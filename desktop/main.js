const { app, BrowserWindow } = require("electron");
const http = require("http");
const net = require("net");
const path = require("path");
const { spawn } = require("child_process");

let backendProcess = null;

function pickPort(preferred) {
  return new Promise((resolve) => {
    const tryPort = (port) => {
      const server = net.createServer();
      server.unref();
      server.on("error", () => tryPort(port + 1));
      server.listen({ host: "127.0.0.1", port }, () => {
        const chosen = server.address().port;
        server.close(() => resolve(chosen));
      });
    };
    tryPort(preferred);
  });
}

function waitForHealth(host, port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  const urlPath = "/health";

  return new Promise((resolve, reject) => {
    const tick = () => {
      if (Date.now() > deadline) {
        reject(new Error("Backend did not start in time"));
        return;
      }

      const req = http.request(
        {
          hostname: host,
          port,
          path: urlPath,
          method: "GET",
          timeout: 1000,
        },
        (res) => {
          res.resume();
          if (res.statusCode === 200) {
            resolve();
            return;
          }
          setTimeout(tick, 250);
        }
      );

      req.on("error", () => setTimeout(tick, 250));
      req.on("timeout", () => {
        req.destroy();
        setTimeout(tick, 250);
      });
      req.end();
    };

    tick();
  });
}

function startBackend({ host, port }) {
  const repoRoot = path.join(__dirname, "..");
  const env = {
    ...process.env,
    HOST: host,
    PORT: String(port),
    RELOAD: "0",
    OPEN_BROWSER: "0",
  };

  const pythonCmd = process.env.PYTHON || "python3";
  backendProcess = spawn(pythonCmd, ["-m", "app"], {
    cwd: repoRoot,
    env,
    stdio: "ignore",
  });

  backendProcess.on("exit", () => {
    backendProcess = null;
  });
}

function stopBackend() {
  if (!backendProcess) return;
  try {
    backendProcess.kill();
  } catch (_) {}
  backendProcess = null;
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 760,
    backgroundColor: "#0b0f1a",
    show: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });

  await win.loadFile(path.join(__dirname, "loading.html"));

  const host = "127.0.0.1";
  const port = await pickPort(8002);
  startBackend({ host, port });
  await waitForHealth(host, port, 20000);
  await win.loadURL(`http://${host}:${port}/`);
}

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  stopBackend();
});

app.whenReady().then(createWindow);
