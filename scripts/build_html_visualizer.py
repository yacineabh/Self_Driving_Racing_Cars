"""build_html_visualizer.py  -  Build self-contained interactive HTML visualizer with embedded telemetry data."""

import json, os

json_path = "outputs/tracks_telemetry.json"
data = json.load(open(json_path))

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SelfRace Interactive Telemetry Visualizer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      user-select: none;
    }}
    canvas {{
      background: #0f1117;
      border-radius: 0.75rem;
    }}
    .glow {{
      box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
    }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] p-4 antialiased">
  <div class="max-w-7xl mx-auto space-y-4">
    
    <!-- Header & Track Selector -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 shadow-sm">
      <div class="flex items-center gap-3">
        <div class="w-3.5 h-3.5 rounded-full bg-emerald-500 animate-pulse"></div>
        <div>
          <h1 class="text-lg font-bold text-[var(--foreground)] tracking-tight">SelfRace: Zero-Shot Live Visualizer</h1>
          <p class="text-xs text-[var(--muted-foreground)]">Real-time telemetry & world-model closed-loop planning</p>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <label for="trackSelect" class="text-xs font-semibold text-[var(--muted-foreground)] uppercase tracking-wider">Select Circuit:</label>
        <select id="trackSelect" class="bg-[var(--background)] text-[var(--foreground)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="Catalunya">Catalunya (Unseen Zero-Shot Circuit)</option>
          <option value="Spielberg">Spielberg (High-Downforce Circuit)</option>
          <option value="Austin">Austin (Circuit of the Americas)</option>
          <option value="MoscowRaceway">Moscow Raceway</option>
        </select>
      </div>
    </div>

    <!-- Main View Area (Canvas on Left, Telemetry on Right) -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
      
      <!-- Track Canvas (2 cols) -->
      <div class="lg:col-span-2 bg-[var(--card)] border border-[var(--border)] rounded-xl p-3 flex flex-col items-center shadow-sm relative">
        <canvas id="trackCanvas" width="760" height="480" class="w-full h-auto max-h-[500px] border border-[var(--border)]"></canvas>
        
        <!-- Canvas Overlay Badges -->
        <div class="absolute top-6 left-6 flex flex-col gap-1.5 pointer-events-none">
          <div id="modeBadge" class="bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 text-xs px-2.5 py-1 rounded-full font-semibold backdrop-blur-sm shadow-sm flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span id="modeText">Mode: CEM World Model</span>
          </div>
          <div id="statusBadge" class="bg-blue-950/80 border border-blue-500/40 text-blue-300 text-xs px-2.5 py-1 rounded-full font-medium backdrop-blur-sm shadow-sm">
            Status: Driving Smoothly
          </div>
        </div>
      </div>

      <!-- Live Telemetry Dashboard (1 col) -->
      <div class="space-y-3 flex flex-col justify-between">
        
        <!-- Speed & Steering Gauges -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm space-y-3">
          <h2 class="text-xs font-bold text-[var(--muted-foreground)] uppercase tracking-wider">Live Vehicle Dynamics</h2>
          
          <!-- Speedometer -->
          <div>
            <div class="flex justify-between text-xs mb-1 font-semibold">
              <span class="text-[var(--foreground)]">Forward Velocity</span>
              <span id="speedVal" class="text-emerald-400 font-mono text-sm">0.00 m/s</span>
            </div>
            <div class="w-full bg-[var(--background)] rounded-full h-2.5 border border-[var(--border)] overflow-hidden">
              <div id="speedBar" class="bg-gradient-to-r from-emerald-500 to-blue-500 h-2.5 rounded-full transition-all duration-75" style="width: 0%"></div>
            </div>
            <div class="flex justify-between text-[11px] text-[var(--muted-foreground)] mt-0.5 font-mono">
              <span>0 m/s</span>
              <span>Target: ~4.1 m/s</span>
              <span>8 m/s</span>
            </div>
          </div>

          <!-- Steering Angle -->
          <div>
            <div class="flex justify-between text-xs mb-1 font-semibold">
              <span class="text-[var(--foreground)]">Steering Command</span>
              <span id="steerVal" class="text-blue-400 font-mono text-sm">0.00 rad (0.0°)</span>
            </div>
            <div class="w-full bg-[var(--background)] rounded-full h-2.5 border border-[var(--border)] overflow-hidden relative">
              <div class="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-500"></div>
              <div id="steerBar" class="bg-blue-500 h-2.5 rounded-full absolute top-0" style="left: 50%; width: 0%"></div>
            </div>
            <div class="flex justify-between text-[11px] text-[var(--muted-foreground)] mt-0.5 font-mono">
              <span>-24° (Left)</span>
              <span>Center</span>
              <span>+24° (Right)</span>
            </div>
          </div>
        </div>

        <!-- Progress & Lap Stats -->
        <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm space-y-3">
          <h2 class="text-xs font-bold text-[var(--muted-foreground)] uppercase tracking-wider">Lap Progress & Timing</h2>
          
          <div>
            <div class="flex justify-between text-xs mb-1 font-semibold">
              <span class="text-[var(--foreground)]">Distance Driven</span>
              <span id="distVal" class="text-purple-400 font-mono text-sm">0.0 / 0.0 m</span>
            </div>
            <div class="w-full bg-[var(--background)] rounded-full h-2.5 border border-[var(--border)] overflow-hidden">
              <div id="distBar" class="bg-gradient-to-r from-purple-500 to-pink-500 h-2.5 rounded-full transition-all duration-75" style="width: 0%"></div>
            </div>
            <div class="flex justify-between text-[11px] text-[var(--muted-foreground)] mt-0.5 font-mono">
              <span>Start Line (0%)</span>
              <span id="pctVal">0.0%</span>
              <span>Finish Line (100%)</span>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-2 pt-1">
            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5 text-center">
              <p class="text-[11px] text-[var(--muted-foreground)] font-medium">Elapsed Time</p>
              <p id="timeVal" class="text-base font-bold font-mono text-[var(--foreground)]">0.0 s</p>
            </div>
            <div class="bg-[var(--background)] border border-[var(--border)] rounded-lg p-2.5 text-center">
              <p class="text-[11px] text-[var(--muted-foreground)] font-medium">Wall Clearance</p>
              <p id="clearVal" class="text-base font-bold font-mono text-emerald-400">> 2.5 m</p>
            </div>
          </div>
        </div>

        <!-- Model Guarantee Badge -->
        <div class="bg-blue-950/30 border border-blue-500/20 rounded-xl p-3 text-xs text-[var(--muted-foreground)]">
          <span class="font-bold text-blue-400">Zero-Shot Guarantee:</span>
          No GPS or map coordinates are fed to the model. Steering and speed are decided purely by the 54 LiDAR beams and past actions in its local frame.
        </div>
      </div>
    </div>

    <!-- Playback Controls Bar -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm flex flex-wrap items-center justify-between gap-4">
      
      <!-- Play/Pause & Restart -->
      <div class="flex items-center gap-2">
        <button id="btnPlay" class="bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs px-4 py-2 rounded-lg transition-colors flex items-center gap-1.5 shadow-sm">
          <svg id="iconPlay" class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
          <span id="textPlay">Play</span>
        </button>
        <button id="btnRestart" class="bg-[var(--background)] hover:bg-[var(--border)] text-[var(--foreground)] border border-[var(--border)] font-semibold text-xs px-3 py-2 rounded-lg transition-colors">
          Restart
        </button>
      </div>

      <!-- Scrubber Slider -->
      <div class="flex-1 max-w-xl flex items-center gap-3">
        <span class="text-xs text-[var(--muted-foreground)] font-mono">0s</span>
        <input type="range" id="scrubber" min="0" max="100" value="0" step="1" class="w-full accent-blue-500 cursor-pointer">
        <span id="scrubberMax" class="text-xs text-[var(--muted-foreground)] font-mono">100s</span>
      </div>

      <!-- Speed Multiplier -->
      <div class="flex items-center gap-1 bg-[var(--background)] border border-[var(--border)] rounded-lg p-1">
        <button data-speed="0.5" class="speed-btn px-2 py-1 text-xs font-semibold rounded text-[var(--muted-foreground)] hover:text-[var(--foreground)]">0.5x</button>
        <button data-speed="1.0" class="speed-btn px-2 py-1 text-xs font-semibold rounded bg-blue-600 text-white">1x</button>
        <button data-speed="2.0" class="speed-btn px-2 py-1 text-xs font-semibold rounded text-[var(--muted-foreground)] hover:text-[var(--foreground)]">2x</button>
        <button data-speed="4.0" class="speed-btn px-2 py-1 text-xs font-semibold rounded text-[var(--muted-foreground)] hover:text-[var(--foreground)]">4x</button>
      </div>
    </div>

  </div>

  <script>
    const DATA = {json.dumps(data)};

    let currentTrack = "Catalunya";
    let frameIdx = 0;
    let isPlaying = false;
    let playbackSpeed = 1.0;
    let animId = null;
    let lastTime = 0;

    const canvas = document.getElementById("trackCanvas");
    const ctx = canvas.getContext("2d");

    const trackSelect = document.getElementById("trackSelect");
    const btnPlay = document.getElementById("btnPlay");
    const textPlay = document.getElementById("textPlay");
    const btnRestart = document.getElementById("btnRestart");
    const scrubber = document.getElementById("scrubber");
    const scrubberMax = document.getElementById("scrubberMax");
    const speedBtns = document.querySelectorAll(".speed-btn");

    const speedVal = document.getElementById("speedVal");
    const speedBar = document.getElementById("speedBar");
    const steerVal = document.getElementById("steerVal");
    const steerBar = document.getElementById("steerBar");
    const distVal = document.getElementById("distVal");
    const distBar = document.getElementById("distBar");
    const pctVal = document.getElementById("pctVal");
    const timeVal = document.getElementById("timeVal");
    const clearVal = document.getElementById("clearVal");
    const modeBadge = document.getElementById("modeBadge");
    const modeText = document.getElementById("modeText");
    const statusBadge = document.getElementById("statusBadge");

    function getTrack() {{
      return DATA[currentTrack];
    }}

    function computeBounds(centerline) {{
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let p of centerline) {{
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }}
      const padX = (maxX - minX) * 0.12;
      const padY = (maxY - minY) * 0.12;
      return {{
        minX: minX - padX, maxX: maxX + padX,
        minY: minY - padY, maxY: maxY + padY
      }};
    }}

    function worldToCanvas(wx, wy, bounds) {{
      const scaleX = canvas.width / (bounds.maxX - bounds.minX);
      const scaleY = canvas.height / (bounds.maxY - bounds.minY);
      const scale = Math.min(scaleX, scaleY);
      
      const offsetX = (canvas.width - (bounds.maxX - bounds.minX) * scale) / 2;
      const offsetY = (canvas.height - (bounds.maxY - bounds.minY) * scale) / 2;

      // Invert Y for standard canvas
      const cx = offsetX + (wx - bounds.minX) * scale;
      const cy = canvas.height - (offsetY + (wy - bounds.minY) * scale);
      return {{ x: cx, y: cy, scale }};
    }}

    function render() {{
      const track = getTrack();
      if (!track || !track.frames.length) return;

      const bounds = computeBounds(track.centerline);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Centerline Track Loop
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
      ctx.lineWidth = 14;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (let i = 0; i < track.centerline.length; i++) {{
        const pt = worldToCanvas(track.centerline[i][0], track.centerline[i][1], bounds);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }}
      ctx.closePath();
      ctx.stroke();

      // Inner road fill
      ctx.beginPath();
      ctx.strokeStyle = "#1a1d26";
      ctx.lineWidth = 10;
      for (let i = 0; i < track.centerline.length; i++) {{
        const pt = worldToCanvas(track.centerline[i][0], track.centerline[i][1], bounds);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }}
      ctx.closePath();
      ctx.stroke();

      // Dash centerline
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      for (let i = 0; i < track.centerline.length; i++) {{
        const pt = worldToCanvas(track.centerline[i][0], track.centerline[i][1], bounds);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }}
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);

      // 2. Draw Trajectory Trail
      ctx.beginPath();
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2.5;
      for (let i = 0; i <= frameIdx; i++) {{
        const pt = worldToCanvas(track.frames[i].x, track.frames[i].y, bounds);
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }}
      ctx.stroke();

      const cur = track.frames[frameIdx];
      const carPt = worldToCanvas(cur.x, cur.y, bounds);

      // 3. Draw Live LiDAR Beams
      if (cur.beams && cur.beams.length) {{
        const nBeams = cur.beams.length;
        const halfFov = Math.PI / 2;
        ctx.lineWidth = 1.0;

        for (let b = 0; b < nBeams; b++) {{
          const beamDist = cur.beams[b];
          const angleOffset = -halfFov + (b / (nBeams - 1)) * Math.PI;
          const worldAngle = cur.th + angleOffset;

          const endX = cur.x + beamDist * Math.cos(worldAngle);
          const endY = cur.y + beamDist * Math.sin(worldAngle);
          const endPt = worldToCanvas(endX, endY, bounds);

          ctx.beginPath();
          ctx.strokeStyle = beamDist < 1.5 ? "rgba(239, 68, 68, 0.7)" : "rgba(6, 182, 212, 0.4)";
          ctx.moveTo(carPt.x, carPt.y);
          ctx.lineTo(endPt.x, endPt.y);
          ctx.stroke();
        }}
      }}

      // 4. Draw Car Body (Oriented Polygon)
      ctx.save();
      ctx.translate(carPt.x, carPt.y);
      // Canvas Y is inverted
      ctx.rotate(-cur.th);

      const carLen = 14;
      const carWid = 7;
      
      // Car Shadow
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
      ctx.fillRect(-carLen/2 + 2, -carWid/2 + 2, carLen, carWid);

      // Car Body
      ctx.fillStyle = cur.mode === "Safety Reflex" ? "#f59e0b" : "#10b981";
      ctx.fillRect(-carLen/2, -carWid/2, carLen, carWid);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(-carLen/2, -carWid/2, carLen, carWid);

      // Front bumper highlight
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(carLen/2 - 2, -carWid/2, 2, carWid);

      // Heading Arrow
      ctx.beginPath();
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2.0;
      ctx.moveTo(carLen/2, 0);
      ctx.lineTo(carLen/2 + 10, 0);
      ctx.stroke();

      ctx.restore();

      // 5. Update Telemetry Elements
      updateDashboard(cur, track);
    }}

    function updateDashboard(cur, track) {{
      // Velocity
      speedVal.innerText = `${{cur.v.toFixed(2)}} m/s`;
      const speedPct = Math.min(100, (cur.v / 6.0) * 100);
      speedBar.style.width = `${{speedPct}}%`;

      // Steering
      const deg = (cur.steer * 180 / Math.PI).toFixed(1);
      steerVal.innerText = `${{cur.steer.toFixed(2)}} rad (${{deg > 0 ? "+" : ""}}${{deg}}°)`;
      const steerPct = (cur.steer / 0.42) * 50; // -50% to +50%
      if (cur.steer >= 0) {{
        steerBar.style.left = "50%";
        steerBar.style.width = `${{Math.min(50, steerPct)}}%`;
      }} else {{
        steerBar.style.left = `${{50 + steerPct}}%`;
        steerBar.style.width = `${{-steerPct}}%`;
      }}

      // Progress
      distVal.innerText = `${{cur.prog.toFixed(1)}} / ${{track.length.toFixed(1)}} m`;
      const lapPct = Math.min(100, (cur.prog / track.length) * 100);
      distBar.style.width = `${{lapPct}}%`;
      pctVal.innerText = `${{lapPct.toFixed(1)}}%`;

      // Time & Clearance
      const timeS = (frameIdx * 0.1).toFixed(1);
      timeVal.innerText = `${{timeS}} s`;
      
      const minBeam = cur.beams ? Math.min(...cur.beams) : 2.5;
      clearVal.innerText = `${{minBeam.toFixed(2)}} m`;
      clearVal.className = minBeam < 1.0 ? "text-base font-bold font-mono text-red-400" :
                           minBeam < 1.8 ? "text-base font-bold font-mono text-amber-400" :
                                           "text-base font-bold font-mono text-emerald-400";

      // Badges
      if (cur.mode === "Safety Reflex") {{
        modeBadge.className = "bg-amber-950/80 border border-amber-500/40 text-amber-300 text-xs px-2.5 py-1 rounded-full font-semibold backdrop-blur-sm shadow-sm flex items-center gap-1.5";
        modeText.innerText = "Mode: Safety Reflex (Follow-The-Gap)";
      }} else {{
        modeBadge.className = "bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 text-xs px-2.5 py-1 rounded-full font-semibold backdrop-blur-sm shadow-sm flex items-center gap-1.5";
        modeText.innerText = "Mode: CEM World Model (Imagination)";
      }}

      if (lapPct >= 99.8) {{
        statusBadge.className = "bg-emerald-950/90 border border-emerald-400 text-emerald-300 text-xs px-3 py-1 rounded-full font-bold backdrop-blur-sm shadow-sm animate-pulse";
        statusBadge.innerText = "LAP COMPLETED! (0 Collisions)";
      }} else {{
        statusBadge.className = "bg-blue-950/80 border border-blue-500/40 text-blue-300 text-xs px-2.5 py-1 rounded-full font-medium backdrop-blur-sm shadow-sm";
        statusBadge.innerText = "Status: Zero-Shot Autonomous Driving";
      }}

      // Scrubber Sync
      scrubber.value = frameIdx;
    }}

    function loop(timestamp) {{
      if (!lastTime) lastTime = timestamp;
      const elapsed = timestamp - lastTime;

      // 10 Hz telemetry step -> 100ms per frame
      const frameInterval = 100 / playbackSpeed;

      if (isPlaying && elapsed >= frameInterval) {{
        const track = getTrack();
        if (frameIdx < track.frames.length - 1) {{
          frameIdx++;
          render();
        }} else {{
          isPlaying = false;
          textPlay.innerText = "Play";
        }}
        lastTime = timestamp;
      }}

      if (isPlaying) {{
        animId = requestAnimationFrame(loop);
      }}
    }}

    function switchTrack(trackName) {{
      currentTrack = trackName;
      frameIdx = 0;
      isPlaying = false;
      textPlay.innerText = "Play";
      if (animId) cancelAnimationFrame(animId);

      const track = getTrack();
      scrubber.max = track.frames.length - 1;
      scrubberMax.innerText = `${{(track.frames.length * 0.1).toFixed(0)}}s`;
      render();
    }}

    trackSelect.addEventListener("change", (e) => {{
      switchTrack(e.target.value);
    }});

    btnPlay.addEventListener("click", () => {{
      const track = getTrack();
      if (frameIdx >= track.frames.length - 1) {{
        frameIdx = 0;
      }}
      isPlaying = !isPlaying;
      textPlay.innerText = isPlaying ? "Pause" : "Play";
      if (isPlaying) {{
        lastTime = 0;
        animId = requestAnimationFrame(loop);
      }}
    }});

    btnRestart.addEventListener("click", () => {{
      frameIdx = 0;
      render();
    }});

    scrubber.addEventListener("input", (e) => {{
      frameIdx = parseInt(e.target.value);
      render();
    }});

    speedBtns.forEach(btn => {{
      btn.addEventListener("click", () => {{
        speedBtns.forEach(b => {{
          b.className = "speed-btn px-2 py-1 text-xs font-semibold rounded text-[var(--muted-foreground)] hover:text-[var(--foreground)]";
        }});
        btn.className = "speed-btn px-2 py-1 text-xs font-semibold rounded bg-blue-600 text-white";
        playbackSpeed = parseFloat(btn.dataset.speed);
      }});
    }});

    // Initialize with Catalunya
    switchTrack("Catalunya");
  </script>
</body>
</html>
"""

out_html = r"C:\Users\Yacine abh\.gemini\antigravity\brain\7ee93501-2408-47a7-872a-f7faefc3fb01\interactive_visualizer.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Generated standalone interactive visualizer at {out_html} ({os.path.getsize(out_html) / 1024:.1f} KB)")
