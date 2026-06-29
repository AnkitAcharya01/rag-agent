from groq import Groq
import os
from dotenv import load_dotenv

import base64, random, uuid, html
import streamlit.components.v1 as components

load_dotenv()

client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)
def tts(text:str):
    response = client.audio.speech.create(
        model= "canopylabs/orpheus-v1-english",
        voice="autumn",
        input=text[:200],
        response_format="wav"
    )

    return response.read()

# def render_assistant_bubble(text: str, audio_bytes: bytes | None = None, autoplay: bool = False):
#     safe_text = html.escape(text)
#     uid = uuid.uuid4().hex[:8]

#     if not audio_bytes:
#         components.html(f'<div style="font-family:sans-serif;font-size:15px;white-space:pre-wrap;">{safe_text}</div>',
#                          height=60, scrolling=True)
#         return

#     b64 = base64.b64encode(audio_bytes).decode()
#     bars = "".join(f'<span style="height:{random.randint(8,22)}px;"></span>' for _ in range(20))

#     html_code = f"""
#     <div style="font-family:sans-serif;font-size:15px;">
#       <div style="display:flex;align-items:flex-end;gap:6px;flex-wrap:wrap;">
#         <span style="line-height:1.4;white-space:pre-wrap;">{safe_text}</span>
#         <button id="icon_{uid}" onclick="toggle_{uid}()"
#                 style="background:none;border:none;cursor:pointer;font-size:16px;padding:0 2px;">
#                 {"⏸" if autoplay else "🔊"}</button>
#       </div>
#       <div id="player_{uid}" style="display:{'flex' if autoplay else 'none'};align-items:center;gap:8px;
#            margin-top:6px;background:#dcf8c6;border-radius:14px;padding:6px 10px;max-width:220px;">
#         <button id="btn_{uid}" onclick="togglePlay_{uid}()"
#                 style="background:#25D366;color:white;border:none;border-radius:50%;
#                        width:28px;height:28px;cursor:pointer;">{"⏸" if autoplay else "▶"}</button>
#         <div id="bars_{uid}" style="display:flex;align-items:center;gap:2px;height:22px;flex:1;">{bars}</div>
#         <audio id="audio_{uid}" src="data:audio/mp3;base64,{b64}" {"autoplay" if autoplay else ""}></audio>
#       </div>
#     </div>
#     <style>
#       #bars_{uid} span {{ width:3px; background:#4a4a4a; border-radius:2px; display:inline-block; }}
#       #bars_{uid}.playing span {{ animation: wave_{uid} 0.9s infinite ease-in-out; }}
#       #bars_{uid}.playing span:nth-child(odd) {{ animation-delay: 0.15s; }}
#       @keyframes wave_{uid} {{ 0%,100% {{ transform: scaleY(0.4); }} 50% {{ transform: scaleY(1); }} }}
#     </style>
#     <script>
#       const player_{uid}=document.getElementById('player_{uid}');
#       const audio_{uid}=document.getElementById('audio_{uid}');
#       const bars_{uid}=document.getElementById('bars_{uid}');
#       const btn_{uid}=document.getElementById('btn_{uid}');
#       const icon_{uid}=document.getElementById('icon_{uid}');
#       let expanded_{uid}={str(autoplay).lower()};

#       function setPlaying_{uid}(p){{
#         bars_{uid}.classList.toggle('playing',p);
#         btn_{uid}.textContent=p?'⏸':'▶';
#         icon_{uid}.textContent=p?'⏸':'🔊';
#       }}
#       function togglePlay_{uid}(){{ audio_{uid}.paused?audio_{uid}.play():audio_{uid}.pause(); }}
#       function toggle_{uid}(){{
#         expanded_{uid}=!expanded_{uid};
#         player_{uid}.style.display=expanded_{uid}?'flex':'none';
#         expanded_{uid}?audio_{uid}.play():audio_{uid}.pause();
#       }}
#       audio_{uid}.onplay=()=>setPlaying_{uid}(true);
#       audio_{uid}.onpause=()=>setPlaying_{uid}(false);
#       audio_{uid}.onended=()=>setPlaying_{uid}(false);
#     </script>
#     """
#     components.html(html_code, height=120 if autoplay else 70, scrolling=True)

# ------------------------------

def render_audio_bubble(audio_bytes: bytes, autoplay: bool = False):
    b64 = base64.b64encode(audio_bytes).decode()
    uid = uuid.uuid4().hex[:8]
    bars = "".join(f'<span style="height:{random.randint(8,20)}px;"></span>' for _ in range(20))
    collapsed_h, expanded_h = 40, 60

    html_code = f"""
    <div id="wrap_{uid}" onclick="toggle_{uid}()"
         style="display:flex;align-items:center;gap:8px;background:#dcf8c6;
                border-radius:14px;padding:6px 10px;max-width:220px;
                font-family:sans-serif;cursor:pointer;">
      <button id="btn_{uid}" style="background:#25D366;color:white;border:none;
              border-radius:50%;width:26px;height:26px;flex-shrink:0;pointer-events:none;">
              {"⏸" if autoplay else "▶"}</button>
      <div id="bars_{uid}" style="display:flex;align-items:center;gap:2px;height:20px;
           overflow:hidden;width:{'110px' if autoplay else '0px'};transition:width .25s ease;">{bars}</div>
      <audio id="audio_{uid}" src="data:audio/mp3;base64,{b64}" {"autoplay" if autoplay else ""}></audio>
    </div>
    <style>
      #bars_{uid} span {{ width:3px;background:#4a4a4a;border-radius:2px;flex-shrink:0;display:inline-block; }}
      #bars_{uid}.playing span {{ animation: wave_{uid} 0.9s infinite ease-in-out; }}
      #bars_{uid}.playing span:nth-child(odd) {{ animation-delay:.15s; }}
      @keyframes wave_{uid} {{ 0%,100% {{ transform:scaleY(0.4); }} 50% {{ transform:scaleY(1); }} }}
    </style>
    <script>
      const audio_{uid}=document.getElementById('audio_{uid}');
      const bars_{uid}=document.getElementById('bars_{uid}');
      const btn_{uid}=document.getElementById('btn_{uid}');
      let expanded_{uid}={str(autoplay).lower()};

      function resize_{uid}(h){{ try{{ window.frameElement.style.height = h+'px'; }}catch(e){{}} }}
      function setPlaying_{uid}(p){{
        bars_{uid}.classList.toggle('playing',p);
        btn_{uid}.textContent=p?'⏸':'▶';
      }}
      function toggle_{uid}(){{
        expanded_{uid}=!expanded_{uid};
        bars_{uid}.style.width = expanded_{uid} ? '110px' : '0px';
        resize_{uid}(expanded_{uid} ? {expanded_h} : {collapsed_h});
        expanded_{uid} ? audio_{uid}.play() : audio_{uid}.pause();
      }}
      audio_{uid}.onplay=()=>setPlaying_{uid}(true);
      audio_{uid}.onpause=()=>setPlaying_{uid}(false);
      audio_{uid}.onended=()=>setPlaying_{uid}(false);
    </script>
    """
    components.html(html_code, height=expanded_h if autoplay else collapsed_h)



    # st.markdown(md, unsafe_allow_html=True)