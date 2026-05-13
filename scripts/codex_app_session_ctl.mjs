#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";

function usage() {
  console.error(`Usage:
  codex_app_session_ctl.mjs create --session-file FILE --url WS_URL [--cwd DIR] [--approval never] [--sandbox workspace-write] [--developer-instructions TEXT]
  codex_app_session_ctl.mjs send --session-file FILE --message TEXT [--wait] [--output-file FILE]
  codex_app_session_ctl.mjs read --session-file FILE [--output-file FILE]
`);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const cur = argv[i];
    if (!cur.startsWith("--")) {
      args._.push(cur);
      continue;
    }
    const key = cur.slice(2);
    if (key === "wait") {
      args.wait = true;
      continue;
    }
    const val = argv[i + 1];
    if (val == null) {
      throw new Error(`missing value for ${cur}`);
    }
    args[key.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = val;
    i += 1;
  }
  return args;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function latestActiveTurnId(thread) {
  const turns = Array.isArray(thread?.turns) ? thread.turns : [];
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    const turn = turns[i];
    if (turn && turn.status === "inProgress") {
      return String(turn.id);
    }
  }
  return null;
}

function latestAgentMessage(thread) {
  const turns = Array.isArray(thread?.turns) ? thread.turns : [];
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    const items = Array.isArray(turns[i]?.items) ? turns[i].items : [];
    for (let j = items.length - 1; j >= 0; j -= 1) {
      const item = items[j];
      if (item && item.type === "agentMessage" && typeof item.text === "string") {
        return item.text;
      }
    }
  }
  return "";
}

class RpcClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.nextId = 1;
    this.pending = new Map();
    this.notifications = [];
  }

  async connect() {
    this.ws = new WebSocket(this.url);
    await new Promise((resolve, reject) => {
      this.ws.addEventListener("open", resolve, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    this.ws.addEventListener("message", (event) => {
      const msg = JSON.parse(String(event.data));
      if (Object.prototype.hasOwnProperty.call(msg, "id")) {
        const key = String(msg.id);
        const entry = this.pending.get(key);
        if (!entry) {
          return;
        }
        this.pending.delete(key);
        if (Object.prototype.hasOwnProperty.call(msg, "error")) {
          entry.reject(new Error(JSON.stringify(msg.error)));
        } else {
          entry.resolve(msg.result);
        }
        return;
      }
      this.notifications.push(msg);
    });
    await this.request("initialize", {
      clientInfo: { name: "codex-app-session-ctl", version: "0.1.0" },
    });
  }

  async request(method, params) {
    const id = String(this.nextId++);
    const payload = { id, method, params };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(payload));
    });
  }

  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

async function waitForTurnCompletion(client, threadId, turnId, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const threadResp = await client.request("thread/read", {
      threadId,
      includeTurns: true,
    });
    const turns = Array.isArray(threadResp?.thread?.turns) ? threadResp.thread.turns : [];
    const turn = turns.find((t) => String(t.id) === String(turnId));
    if (turn && turn.status !== "inProgress") {
      return threadResp.thread;
    }
    await sleep(300);
  }
  throw new Error(`timed out waiting for turn ${turnId} to finish`);
}

async function cmdCreate(args) {
  if (!args.sessionFile || !args.url) {
    throw new Error("create requires --session-file and --url");
  }
  const client = new RpcClient(args.url);
  await client.connect();
  try {
    const resp = await client.request("thread/start", {
      cwd: args.cwd || process.cwd(),
      approvalPolicy: args.approval || "never",
      sandbox: args.sandbox || "workspace-write",
      developerInstructions: args.developerInstructions || null,
    });
    const session = {
      url: args.url,
      threadId: resp.thread.id,
      cwd: args.cwd || process.cwd(),
      createdAt: new Date().toISOString(),
      developerInstructions: args.developerInstructions || "",
    };
    mkdirSync(path.dirname(args.sessionFile), { recursive: true });
    writeFileSync(args.sessionFile, `${JSON.stringify(session, null, 2)}\n`);
    console.log(JSON.stringify(session, null, 2));
  } finally {
    client.close();
  }
}

async function cmdSend(args) {
  if (!args.sessionFile || !args.message) {
    throw new Error("send requires --session-file and --message");
  }
  const session = JSON.parse(readFileSync(args.sessionFile, "utf8"));
  const client = new RpcClient(session.url);
  await client.connect();
  try {
    let threadResp = null;
    let activeTurnId = null;
    try {
      threadResp = await client.request("thread/read", {
        threadId: session.threadId,
        includeTurns: true,
      });
      activeTurnId = latestActiveTurnId(threadResp.thread);
    } catch (err) {
      const msg = String(err?.message || err);
      if (!msg.includes("includeTurns is unavailable before first user message")) {
        throw err;
      }
    }
    let mode;
    let turnId;
    if (activeTurnId) {
      mode = "steer";
      try {
        const steerResp = await client.request("turn/steer", {
          threadId: session.threadId,
          expectedTurnId: activeTurnId,
          input: [{ type: "text", text: args.message }],
        });
        turnId = steerResp.turnId;
      } catch (err) {
        if (!String(err.message || err).includes("no active turn")) {
          throw err;
        }
        mode = "start";
        const startResp = await client.request("turn/start", {
          threadId: session.threadId,
          input: [{ type: "text", text: args.message }],
        });
        turnId = startResp.turn.id;
      }
    } else {
      mode = "start";
      const startResp = await client.request("turn/start", {
        threadId: session.threadId,
        input: [{ type: "text", text: args.message }],
      });
      turnId = startResp.turn.id;
    }

    let result = {
      mode,
      threadId: session.threadId,
      turnId,
      message: args.message,
    };

    if (args.wait) {
      const finalThread = await waitForTurnCompletion(client, session.threadId, turnId);
      result.finalAgentMessage = latestAgentMessage(finalThread);
      result.threadStatus = finalThread.status;
    }

    const out = JSON.stringify(result, null, 2);
    if (args.outputFile) {
      mkdirSync(path.dirname(args.outputFile), { recursive: true });
      writeFileSync(args.outputFile, `${out}\n`);
    }
    console.log(out);
  } finally {
    client.close();
  }
}

async function cmdRead(args) {
  if (!args.sessionFile) {
    throw new Error("read requires --session-file");
  }
  const session = JSON.parse(readFileSync(args.sessionFile, "utf8"));
  const client = new RpcClient(session.url);
  await client.connect();
  try {
    const threadResp = await client.request("thread/read", {
      threadId: session.threadId,
      includeTurns: true,
    });
    const out = JSON.stringify(threadResp, null, 2);
    if (args.outputFile) {
      mkdirSync(path.dirname(args.outputFile), { recursive: true });
      writeFileSync(args.outputFile, `${out}\n`);
    }
    console.log(out);
  } finally {
    client.close();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cmd = args._[0];
  if (!cmd) {
    usage();
    process.exit(2);
  }
  if (cmd === "create") {
    await cmdCreate(args);
    return;
  }
  if (cmd === "send") {
    await cmdSend(args);
    return;
  }
  if (cmd === "read") {
    await cmdRead(args);
    return;
  }
  throw new Error(`unknown command: ${cmd}`);
}

main().catch((err) => {
  console.error(err.stack || String(err));
  process.exit(1);
});
