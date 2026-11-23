import { createContextMonitor } from "./context-window.js";

const form = document.querySelector("#composer");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const newChat = document.querySelector("#new-chat");
const status = document.querySelector("#chat-status");
const transcript = document.querySelector("#transcript");
const showUsage = createContextMonitor(document.querySelector("#context-window"));
let conversationId = null;

function appendMessage(role, text) {
  const message = document.createElement("article");
  const label = document.createElement("h2");
  const content = document.createElement("p");
  label.textContent = role;
  content.textContent = text;
  message.append(label, content);
  transcript.append(message);
  transcript.scrollTop = transcript.scrollHeight;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || send.disabled) return;

  send.disabled = true;
  newChat.disabled = true;
  input.readOnly = true;
  status.textContent = "Sending…";
  try {
    const payload = { message };
    if (conversationId) payload.conversation_id = conversationId;
    const response = await fetch("/chat?include_usage=true", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      status.textContent = typeof result.detail === "string" ? result.detail : "Chat request failed.";
      showUsage(null);
      return;
    }
    conversationId = result.conversation_id;
    appendMessage("You", message);
    appendMessage("Assistant", result.answer);
    showUsage(result.usage);
    input.value = "";
    status.textContent = "";
  } catch {
    status.textContent = "Could not reach the chat service.";
    showUsage(null);
  } finally {
    send.disabled = false;
    newChat.disabled = false;
    input.readOnly = false;
    input.focus();
  }
});

newChat.addEventListener("click", () => {
  conversationId = null;
  transcript.replaceChildren();
  showUsage(null);
  status.textContent = "";
  input.value = "";
  input.focus();
});
