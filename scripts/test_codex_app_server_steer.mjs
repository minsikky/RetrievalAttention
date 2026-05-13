#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createWriteStream, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";

const workdir = process.cwd();
const outDir = path.join(workdir, ".codex", "app_server_test");
mkdirSync(outDir, { recursive: true });

const port = Number(process.env.CODEX_APP_SERVER_TEST_PORT || 8788);
const url = `ws://127.0.0.1:${port}`;
const steerToken = `STEER_OK_${Date.now()}`;
const originalToken = `ORIGINAL_APP_SERVER_${Date.now()}`;
const prompt = [
  "RUN_SHELL exactly once and do not answer before it finishes.",
  "Use the shell tool to run exactly this command:",
  `bash -lc 'sleep 8; printf ${originalToken}'`,
  `After it finishes, reply with exactly ${originalToken} and nothing else.`,
].join("\n");
const steerPrompt = `Change course. After the ongoing work finishes, reply with exactly ${steerToken} and nothing else.`;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function waitForPort(host, portNum, timeoutMs = 10000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const sock = net.connect({ host, port: portNum });
      sock.once("connect", () => {
        sock.destroy();
        resolve();
      });
      sock.once("error", () => {
        sock.destroy();
        if (Date.now() - start >= timeoutMs) {
          reject(new Error(`timed out waiting for ${host}:${portNum}`));
        } else {
          setTimeout(tryOnce, 200);
        }
      });
    };
    tryOnce();
  });
}

async function main() {
  const stdoutLog = createWriteStream(path.join(outDir, "app_server_stdout.log"), { flags: "a" });
  const stderrLog = createWriteStream(path.join(outDir, "app_server_stderr.log"), { flags: "a" });

  const server = spawn(
    "codex",
    ["app-server", "--listen", url],
    { cwd: workdir, stdio: ["ignore", "pipe", "pipe"] },
  );
  server.stdout.pipe(stdoutLog);
  server.stderr.pipe(stderrLog);

  let closed = false;
  const closeServer = () => {
    if (closed) return;
    closed = true;
    server.kill("SIGTERM");
  };
  process.on("exit", closeServer);
  process.on("SIGINT", () => {
    closeServer();
    process.exit(130);
  });

  await waitForPort("127.0.0.1", port, 15000);

  const ws = new WebSocket(url);
  const pending = new Map();
  const notifications = [];
  let completedTurn = null;
  let steerAccepted = false;
  let steerAttempts = 0;
  let lastSteerError = null;
  let completedResolve;
  let completedReject;
  const completedPromise = new Promise((resolve, reject) => {
    completedResolve = resolve;
    completedReject = reject;
  });
  let nextId = 1;

  const ready = new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", reject, { once: true });
  });

  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(String(event.data));
    if (Object.prototype.hasOwnProperty.call(msg, "id")) {
      const entry = pending.get(String(msg.id));
      if (!entry) return;
      pending.delete(String(msg.id));
      if (Object.prototype.hasOwnProperty.call(msg, "error")) {
        entry.reject(new Error(JSON.stringify(msg.error)));
      } else {
        entry.resolve(msg.result);
      }
      return;
    }
    notifications.push(msg);
    if (msg.method === "error") {
      completedReject(new Error(JSON.stringify(msg.params)));
      return;
    }
    if (msg.method === "turn/completed") {
      completedTurn = msg.params?.turn ?? null;
      completedResolve(completedTurn);
    }
  });

  function request(method, params) {
    const id = String(nextId++);
    const payload = { id, method, params };
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify(payload));
    });
  }

  await ready;

  await request("initialize", {
    clientInfo: { name: "codex-app-server-steer-test", version: "0.1.0" },
  });

  const threadStart = await request("thread/start", {
    cwd: workdir,
    approvalPolicy: "never",
    sandbox: "workspace-write",
    developerInstructions: "When the user asks to RUN_SHELL, you must execute the requested shell command verbatim with the shell tool before answering. Do not skip the command.",
  });
  const threadId = threadStart.thread.id;

  const turnStart = await request("turn/start", {
    threadId,
    input: [{ type: "text", text: prompt }],
  });
  const turnId = turnStart.turn.id;

  async function steerLoop() {
    while (!steerAccepted && !completedTurn) {
      steerAttempts += 1;
      try {
        await request("turn/steer", {
          threadId,
          expectedTurnId: turnId,
          input: [{ type: "text", text: steerPrompt }],
        });
        steerAccepted = true;
        return;
      } catch (err) {
        lastSteerError = String(err?.message || err);
        if (!lastSteerError.includes("no active turn")) {
          throw err;
        }
        await sleep(100);
      }
    }
  }
  const steerPromise = steerLoop();

  const timeout = setTimeout(() => {
    completedReject(new Error("timed out waiting for turn/completed"));
  }, 90000);

  await completedPromise;
  await steerPromise.catch((err) => {
    throw err;
  });
  clearTimeout(timeout);

  const threadRead = await request("thread/read", {
    threadId,
    includeTurns: true,
  });

  const turns = threadRead.thread.turns || [];
  const targetTurn = turns.find((turn) => turn.id === turnId) || turns[turns.length - 1] || null;
  const agentMessages = [];
  if (targetTurn && Array.isArray(targetTurn.items)) {
    for (const item of targetTurn.items) {
      if (item && item.type === "agentMessage" && typeof item.text === "string") {
        agentMessages.push(item.text);
      }
    }
  }
  const finalAgentText = agentMessages.length > 0 ? agentMessages[agentMessages.length - 1] : "";
  const passed = finalAgentText.includes(steerToken);

  const summary = {
    url,
    threadId,
    turnId,
    originalToken,
    steerToken,
    finalAgentText,
    passed,
    steerAccepted,
    steerAttempts,
    lastSteerError,
    notificationMethods: notifications.map((x) => x.method),
    completedTurn,
  };
  const outPath = path.join(outDir, "steer_test_result.json");
  writeFileSync(outPath, JSON.stringify(summary, null, 2));

  console.log(JSON.stringify(summary, null, 2));

  ws.close();
  closeServer();

  if (!passed) {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  const outPath = path.join(outDir, "steer_test_error.txt");
  writeFileSync(outPath, `${err.stack || err}\n`);
  console.error(err);
  process.exit(1);
});
