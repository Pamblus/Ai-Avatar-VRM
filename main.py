#!/usr/bin/env python3
"""Сервер: чат с LLM + раздача BVH + парсер шагов.

LLM получает:
  • свой характер и правила (prompts.txt)
  • ВЕСЬ список .bvh файлов из ANIM_DIR (без обрезки)

Возможности в ответе LLM:
  [pose dur=0.5 hold=1.0]{...}[/pose]     — параллельная поза
  [body ...]{...}[/body]                  — трансформ модели
  [face ...]{...}[/face]                  — морфы лица
  [emote]wave[/emote]                     — пресет
  [wait]0.4[/wait]                        — пауза
  [loop count=N]...[/loop]                — цикл
  [bvh]имя_файла.bvh[/bvh]                — запуск BVH-анимации
  [stop]                                  — вернуться в нейтраль
"""
import json
import os
import re
import socket
import sys
from pathlib import Path

from aiohttp import web, ClientSession, ClientTimeout
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════
#  НАСТРОЙКИ
# ══════════════════════════════════════════════════════
API_KEY         = os.getenv("OPENROUTER_API_KEY")
MODEL           = os.getenv("VRM_MODEL", "openai/gpt...")
MAX_TOKENS      = int(os.getenv("VRM_MAX_TOKENS", "2000"))
TEMPERATURE     = float(os.getenv("VRM_TEMPERATURE", "0.8"))
SITE_URL        = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
SITE_NAME       = os.getenv("OPENROUTER_SITE_NAME", "VRM-AI-Show")
HISTORY_LIMIT   = int(os.getenv("VRM_HISTORY", "20"))
PREFERRED_PORT  = int(os.getenv("PORT", "8000"))
PORT_SCAN_LIMIT = int(os.getenv("PORT_SCAN_LIMIT", "50"))
REQUEST_TIMEOUT = int(os.getenv("VRM_TIMEOUT", "180"))

ENDPOINT    = "https://openrouter.ai/api/v1/chat/completions"
ROOT        = Path(__file__).parent.resolve()
PROMPT_FILE = ROOT / "prompts.txt"
INDEX_FILE  = ROOT / "index.html"

_default_anim = Path.home() / "ai" / "animation"
ANIM_DIR = Path(os.getenv("ANIM_DIR", str(_default_anim))).resolve()

if not API_KEY:
    print("⚠️  OPENROUTER_API_KEY не задан в .env", file=sys.stderr)


# ══════════════════════════════════════════════════════
#  ПОРТЫ
# ══════════════════════════════════════════════════════
def _port_free(port, host="0.0.0.0"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port)); return True
        except OSError:
            return False


def find_free_port(preferred=8000, limit=50, host="0.0.0.0"):
    for off in range(limit):
        p = preferred + off
        if p > 65535: break
        if _port_free(p, host): return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0)); return s.getsockname()[1]


def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ══════════════════════════════════════════════════════
#  КЭШ СПИСКА BVH
# ══════════════════════════════════════════════════════
_anim_cache = {"mtime": 0, "files": []}

def get_animations():
    """Возвращает отсортированный список имён .bvh (с кэшем по mtime)."""
    global _anim_cache
    if not ANIM_DIR.exists():
        return []
    try:
        mtime = ANIM_DIR.stat().st_mtime
        if mtime != _anim_cache["mtime"]:
            _anim_cache["files"] = sorted(
                p.name for p in ANIM_DIR.iterdir()
                if p.is_file() and p.suffix.lower() == ".bvh"
            )
            _anim_cache["mtime"] = mtime
        return _anim_cache["files"]
    except Exception:
        return []


def build_anim_block():
    """Формирует текстовый блок со ВСЕМ списком анимаций для промпта."""
    files = get_animations()
    if not files:
        return ""

    # один файл на строку — так LLM легче копировать точное имя
    lines = "\n".join(f"  - {name}" for name in files)

    return f"""


  БИБЛИОТЕКА АНИМАЦИЙ ({len(files)} файлов)

В основном используй только готовые анимации с помощью  [bvh]name.bvh[/bvh]
У тебя есть {len(files)} BVH-анимаций. Ты можешь запустить любую из них
тегом [bvh]имя_файла.bvh[/bvh]. Когда пользователь просит какое-то
движение/танец/позу/действие — посмотри список ниже, найди наиболее
подходящий файл, и запусти его.

ПРАВИЛА РАБОТЫ С BVH:
1. Имя указывай ТОЧНО как в списке — с расширением .bvh, без опечаток.
2. Если пользователь просит действие, которого явно нет — импровизируй
   через [pose]/[emote], не выдумывай имя файла.
3. Можешь комбинировать: [bvh]одна.bvh[/bvh], [wait]1[/wait],
   [bvh]другая.bvh[/bvh].
4. 
5. Если BVH уже играет, а пользователь просит новое — просто вызови
   новый [bvh], предыдущий остановится автоматически.
6. Для возврата в нейтраль — [stop].

ПОЛНЫЙ СПИСОК ДОСТУПНЫХ ФАЙЛОВ:
{lines}

ПРИМЕРЫ:

Пользователь: «станцуй»
Ты: Сейчас покажу! 💃
[bvh]name.bvh[/bvh]

Пользователь: «помаши рукой и улыбнись»
Ты: Лови! 👋
[bvh]name.bvh[/bvh]

Пользователь: «сделай что-нибудь странное»
Ты: Хе-хе, сейчас будет весело ✨
[bvh]какой-то-файл.bvh[/bvh]

Пользователь: «...»
Ты: Слушаюсь 🎀
[bvh]подходящий.bvh[/bvh]
"""


# ══════════════════════════════════════════════════════
#  РЕГЕКСЫ ПАРСЕРА
# ══════════════════════════════════════════════════════
ATTR_RE  = re.compile(r'(\w+)=("[^"]*"|[\d.]+|forever)')
LOOP_RE  = re.compile(r"\[loop\s+count=(\d+)\](.*?)\[/loop\]", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

TOKEN_RE = re.compile(
    r"\[(pose|step|emote|wait|say|body|face|bvh)"
    r"((?:\s+\w+=(?:\"[^\"]*\"|[\d.]+|forever))*)"
    r"\](.*?)\[/\1\]"
    r"|\[(freeze|stop)\]",
    re.DOTALL | re.IGNORECASE,
)


def _norm_bone(v):
    try:
        if isinstance(v, list) and len(v) == 3:
            return [float(x) for x in v]
        if isinstance(v, dict):
            return [float(v.get("x",0)), float(v.get("y",0)), float(v.get("z",0))]
    except (TypeError, ValueError):
        pass
    return None


def _parse_attrs(s):
    out = {}
    if not s: return out
    for m in ATTR_RE.finditer(s):
        k, v = m.group(1), m.group(2)
        if v == "forever":
            out[k] = "forever"
        else:
            try: out[k] = float(v.strip('"'))
            except ValueError: out[k] = v.strip('"')
    return out


def _json_load(raw):
    fence = FENCE_RE.search(raw)
    if fence: raw = fence.group(1)
    try: return json.loads(raw.strip())
    except Exception: return None


def _json_bones(raw):
    obj = _json_load(raw)
    if not isinstance(obj, dict): return None
    if "bones" in obj and isinstance(obj["bones"], dict):
        obj = obj["bones"]
    out = {}
    for k, v in obj.items():
        nb = _norm_bone(v)
        if nb: out[k] = nb
    return out or None


def parse_inner(text):
    steps, say_parts = [], []
    last_end = 0

    for m in TOKEN_RE.finditer(text):
        between = text[last_end:m.start()].strip()
        if between: say_parts.append(between)
        last_end = m.end()

        # без-теговые [freeze]/[stop]
        if m.group(4):
            tag = m.group(4).lower()
            if tag == "freeze": steps.append({"freeze": True})
            elif tag == "stop": steps.append({"stop": True})
            continue

        tag    = m.group(1).lower()
        attrs  = _parse_attrs(m.group(2))
        body   = (m.group(3) or "").strip()

        dur  = attrs.get("dur")
        hold = attrs.get("hold")
        loop = attrs.get("loop")
        try: loop = int(loop) if loop not in (None,"forever") else None
        except (TypeError, ValueError): loop = None

        # POSE / STEP
        if tag in ("pose","step"):
            obj = None
            fence = FENCE_RE.search(body)
            raw_json = fence.group(1) if fence else body
            try: obj = json.loads(raw_json)
            except Exception: obj = None

            if isinstance(obj, dict) and "wait" in obj:
                try: steps.append({"wait": float(obj["wait"])})
                except: pass
                continue
            if isinstance(obj, dict) and "emote" in obj:
                steps.append({"emote": str(obj["emote"]),
                              "dur": float(obj.get("dur", dur or 1.2))})
                continue
            if isinstance(obj, dict) and "bvh" in obj:
                steps.append({"bvh": str(obj["bvh"])})
                continue

            bones = _json_bones(body)
            if bones:
                s = {"bones": bones,
                     "dur": dur if dur is not None else 0.5,
                     "hold": hold if hold is not None else 1.0}
                if hold == "forever":
                    s["hold"] = "forever"; s["freeze"] = True
                if loop and loop > 1:
                    s["loop"] = loop
                steps.append(s)
            continue

        # BODY
        if tag == "body":
            b = {}
            for k in ("x","y","z","rx","ry","rz"):
                if k in attrs:
                    try: b[k] = float(attrs[k])
                    except: pass
            if b:
                steps.append({"body": b, "dur": dur if dur is not None else 0.8})
            continue

        # FACE
        if tag == "face":
            f = {}
            for k, v in attrs.items():
                if k == "dur": continue
                try: f[k] = float(v)
                except: pass
            if f:
                steps.append({"face": f, "dur": dur if dur is not None else 0.3})
            continue

        # EMOTE
        if tag == "emote":
            name = body.strip().lower()
            if name:
                steps.append({"emote": name,
                              "dur": float(dur) if dur is not None else 1.2})
            continue

        # BVH
        if tag == "bvh":
            name = body.strip()
            if name:
                # защита от лишних пробелов и кавычек
                name = name.strip('"\'')
                steps.append({"bvh": name})
            continue

        # WAIT
        if tag == "wait":
            try: steps.append({"wait": float(body)})
            except ValueError: pass
            continue

        # SAY
        if tag == "say":
            if body: say_parts.append(body)
            continue

    tail = text[last_end:].strip()
    if tail: say_parts.append(tail)

    return steps, say_parts


def parse_ai_response(text):
    loop_ranges = list(LOOP_RE.finditer(text))
    if not loop_ranges:
        steps, say_parts = parse_inner(text)
        return {"say": "\n\n".join(say_parts).strip() or None, "steps": steps}

    steps_out, say_parts = [], []
    pos = 0
    for m in loop_ranges:
        if m.start() > pos:
            s, sp = parse_inner(text[pos:m.start()])
            steps_out.extend(s); say_parts.extend(sp)
        inner = parse_ai_response(m.group(2))
        if inner["say"]: say_parts.append(inner["say"])
        steps_out.append({"loop": int(m.group(1)), "steps": inner["steps"]})
        pos = m.end()
    if pos < len(text):
        s, sp = parse_inner(text[pos:])
        steps_out.extend(s); say_parts.extend(sp)

    return {"say": "\n\n".join(say_parts).strip() or None, "steps": steps_out}


# ══════════════════════════════════════════════════════
#  OPENROUTER
# ══════════════════════════════════════════════════════
async def call_llm(session, messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-OpenRouter-Title": SITE_NAME,
    }
    async with session.post(ENDPOINT, headers=headers, json=payload,
                            timeout=ClientTimeout(total=REQUEST_TIMEOUT)) as r:
        data = await r.json()
        if r.status != 200:
            raise RuntimeError(f"OpenRouter {r.status}: {data}")
        return data["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════════
#  HANDLERS
# ══════════════════════════════════════════════════════
async def handle_chat(request):
    try: body = await request.json()
    except Exception: return web.json_response({"error": "invalid json"}, status=400)

    user_msgs = body.get("messages", [])
    if not isinstance(user_msgs, list):
        return web.json_response({"error": "messages must be array"}, status=400)

    # базовый промпт (характер, правила) + список анимаций
    system_prompt = ""
    if PROMPT_FILE.exists():
        system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    system_prompt += build_anim_block()

    convo = [{"role": "system", "content": system_prompt}]
    convo += user_msgs[-HISTORY_LIMIT:]

    try:
        async with ClientSession() as session:
            raw = await call_llm(session, convo)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

    parsed = parse_ai_response(raw)
    parsed["_raw"] = raw
    return web.json_response(parsed)


async def handle_health(request):
    return web.json_response({
        "ok": True,
        "model": MODEL,
        "port": request.app["port"],
        "prompt_exists": PROMPT_FILE.exists(),
        "index_exists": INDEX_FILE.exists(),
        "anim_dir": str(ANIM_DIR),
        "anim_dir_exists": ANIM_DIR.exists(),
        "anim_count": len(get_animations()),
    })


async def handle_anim_list(request):
    if not ANIM_DIR.exists():
        return web.json_response({"files": [], "error": f"папка не найдена: {ANIM_DIR}"}, status=404)
    try:
        files = get_animations()
    except Exception as e:
        return web.json_response({"files": [], "error": str(e)}, status=500)
    return web.json_response({"files": files, "count": len(files), "dir": str(ANIM_DIR)})


async def handle_anim_reload(request):
    """Принудительно сбросить кэш списка анимаций."""
    _anim_cache["mtime"] = 0
    files = get_animations()
    return web.json_response({"ok": True, "count": len(files)})


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    port = find_free_port(PREFERRED_PORT, PORT_SCAN_LIMIT)
    ip   = get_local_ip()

    anims = get_animations()

    app = web.Application()
    app.router.add_get("/api/health", handle_health)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/animations", handle_anim_list)
    app.router.add_post("/api/animations/reload", handle_anim_reload)
    if ANIM_DIR.exists():
        app.router.add_static("/animations/", ANIM_DIR, show_index=False)
    app.router.add_static("/", ROOT, show_index=True)
    app["port"] = port

    print("═" * 62)
    print("🌸  AI Show — VRM + LLM + BVH (full library)")
    print("═" * 62)
    print(f"  Модель LLM:   {MODEL}")
    print(f"  Промпт:       {PROMPT_FILE.name} " +
          ("✓" if PROMPT_FILE.exists() else "✗"))
    print(f"  Сцена:        {INDEX_FILE.name} " +
          ("✓" if INDEX_FILE.exists() else "✗"))
    if ANIM_DIR.exists():
        print(f"  BVH папка:    {ANIM_DIR}")
        print(f"                ✓ {len(anims)} .bvh → идут в промпт целиком")
    else:
        print(f"  BVH папка:    {ANIM_DIR}  ✗ не найдена")
    if port != PREFERRED_PORT:
        print(f"  ⚠️  Порт {PREFERRED_PORT} занят → {port}")
    print()
    print(f"  ▶ http://localhost:{port}")
    print(f"  ▶ http://{ip}:{port}")
    print(f"  ▶ health: http://localhost:{port}/api/health")
    print("═" * 62)
    sys.stdout.flush()

    try:
        web.run_app(app, host="0.0.0.0", port=port, print=None)
    except OSError as e:
        port = find_free_port(port + 1, PORT_SCAN_LIMIT)
        app["port"] = port
        print(f"\n⚠️  Перезапуск на порту {port}: http://{ip}:{port}\n")
        web.run_app(app, host="0.0.0.0", port=port, print=None)


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n👋 Стоп")
