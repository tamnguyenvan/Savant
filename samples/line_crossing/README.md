# Line crossing demo

The pipeline detects when people cross a user-configured line and the direction of the crossing. The crossing events are attached to individual tracks, counted for each source separately and the counters are displayed on the frame. The crossing events are also stored with Graphite and displayed on a Graphana dashboard.

Preview:

![](../line-crossing-loop.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);

Demonstrated adapters:
- RTSP source adapter;
- Always-ON RTSP sink adapter;

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/line_crossing
git lfs pull

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/line_crossing.html

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:8554/town-centre-processed' in your player
# or visit 'http://127.0.0.1:8888/town-centre-processed/' (LL-HLS)

# for pre-configured Graphana dashboard visit
# http://127.0.0.1:3000/d/WM6WimE4z/entries-exits?orgId=1&refresh=5s

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

To create a custom Graphana dashboard, sign in with `admin\admin` credentials.