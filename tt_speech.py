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


def render_audio_bubble(audio_bytes: bytes, autoplay: bool = False):
    b64 = base64.b64encode(audio_bytes).decode()
    uid = uuid.uuid4().hex[:8]
    bars = "".join(f'<span style="height:{random.randint(8,20)}px;"></span>' for _ in range(20))
    collapsed_h, expanded_h = 48, 60

    html_code = f"""
    <div id="wrap_{uid}" onclick="toggle_{uid}()"
         style="display:inline-flex;
                align-items:center;
                gap:{'8px' if autoplay else '0px'};
                background:#dcf8c6;
                border-radius:14px;
                padding:6px 10px;
                overflow:hidden;
                white-space: nowrap;
                width:{'160px' if autoplay else '42px'};
                transition:width .25s ease, gap .25s ease;
                box-sizing: border-box;                 
                cursor:pointer;">

      <button id="btn_{uid}" style="background:#25D366;
                                    color:white;
                                    border:none;
                                    border-radius:50%;
                                    width:26px;
                                    height:26px;
                                    flex-shrink:0;
                                    pointer-events:none;">

              {"⏸" if autoplay else "▶"}</button>

      <div id="bars_{uid}" style="display:flex;
                           align-items:center;
                           gap:2px;
                           height:20px;
                           overflow:hidden;
                           width:{'110px' if autoplay else '0px'};
                           transition:width .25s ease;">
                          {bars}</div>
                          
      <audio id="audio_{uid}" src="data:audio/mp3;base64,{b64}" {"autoplay" if autoplay else ""}></audio>
    </div>
    <style>
      #bars_{uid} span {{ width:3px;
                          background:#4a4a4a;
                          border-radius:2px;
                          flex-shrink:0;
                          display:inline-block; }}

      #bars_{uid}.playing span {{ animation: wave_{uid} 0.9s infinite ease-in-out; }}
      #bars_{uid}.playing span:nth-child(odd) {{ animation-delay:.15s; }}
      @keyframes wave_{uid} {{ 0%,100% {{ transform:scaleY(0.4); }} 50% {{ transform:scaleY(1); }} }}
      
    </style>
    <script>
      const audio_{uid} = document.getElementById("audio_{uid}");
      const bars_{uid}  = document.getElementById("bars_{uid}");
      const btn_{uid}   = document.getElementById("btn_{uid}");
      const wrap_{uid}  = document.getElementById("wrap_{uid}");

      let expanded_{uid} = {str(autoplay).lower()};

      function resize_{uid}(h) {{
          try {{
              window.frameElement.style.height = h + "px";
          }} catch (e) {{}}
      }}

      function setPlaying_{uid}(playing) {{
          bars_{uid}.classList.toggle("playing", playing);
          btn_{uid}.textContent = playing ? "⏸" : "▶";
      }}

      function expand_{uid}() {{
          expanded_{uid} = true;
          wrap_{uid}.style.width = "160px";
          wrap_{uid}.style.gap = "8px";
          bars_{uid}.style.width = "110px";
          resize_{uid}({expanded_h});
      }}

      function collapse_{uid}() {{
          expanded_{uid} = false;
          wrap_{uid}.style.width = "42px";
          wrap_{uid}.style.gap = "0px";
          bars_{uid}.style.width = "0px";
          resize_{uid}({collapsed_h});
      }}

      async function toggle_{uid}() {{
        if(audio_{uid}.paused) {{
          expand_{uid}();
          try {{
            await audio_{uid}.play();
          }} catch (e) {{
              collapse_{uid}();
          }}
        }} else {{
          audio_{uid}.pause();
        }}
      }}

      audio_{uid}.addEventListener("play", () => {{
          setPlaying_{uid}(true);
          expand_{uid}();
      }});

      audio_{uid}.addEventListener("pause", () => {{
          setPlaying_{uid}(false);
      }});

      audio_{uid}.addEventListener("ended", () => {{
          setPlaying_{uid}(false);
          collapse_{uid}();
      }});
    </script>
    """
    components.html(html_code, height=expanded_h if autoplay else collapsed_h)


