"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const MATERIAL_SYSTEMS = ["perovskite", "MXene", "MOF", "zeolite", "alloy", "other"];
const SIMULATION_SOFTWARE = ["VASP", "LAMMPS", "Quantum ESPRESSO", "GROMACS", "general"];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [materialSystem, setMaterialSystem] = useState("");
  const [simulationSoftware, setSimulationSoftware] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage.content,
          material_system: materialSystem || null,
          simulation_software: simulationSoftware || null,
        }),
      });

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.response || "No response received.",
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: "assistant",
        content: "Error: Could not connect to ChemMind backend.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-chem-dark">
      {/* Sidebar */}
      <aside className="w-64 bg-chem-panel border-r border-chem-border flex flex-col">
        <div className="p-4 border-b border-chem-border">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-chem-primary rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">C</span>
            </div>
            <h1 className="text-lg font-bold text-chem-primary">ChemMind</h1>
          </div>
          <p className="text-xs text-chem-text-muted mt-1">AI for Computational Chemistry</p>
        </div>

        <div className="p-4 space-y-4 flex-1">
          <div>
            <label className="block text-sm font-medium text-chem-text-muted mb-2">
              Material System
            </label>
            <select
              value={materialSystem}
              onChange={(e) => setMaterialSystem(e.target.value)}
              className="w-full bg-chem-dark border border-chem-border rounded-lg px-3 py-2 text-sm text-chem-text focus:outline-none focus:border-chem-primary"
            >
              <option value="">Any</option>
              {MATERIAL_SYSTEMS.map((m) => (
                <option key={m} value={m}>
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-chem-text-muted mb-2">
              Simulation Software
            </label>
            <select
              value={simulationSoftware}
              onChange={(e) => setSimulationSoftware(e.target.value)}
              className="w-full bg-chem-dark border border-chem-border rounded-lg px-3 py-2 text-sm text-chem-text focus:outline-none focus:border-chem-primary"
            >
              <option value="">Any</option>
              {SIMULATION_SOFTWARE.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="p-4 border-t border-chem-border">
          <p className="text-xs text-chem-text-muted">
            Filters apply context to AI responses
          </p>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="w-16 h-16 bg-chem-primaryDark rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-chem-primary text-2xl font-bold">C</span>
                </div>
                <h2 className="text-xl font-semibold text-chem-text mb-2">Welcome to ChemMind</h2>
                <p className="text-chem-text-muted max-w-md">
                  Ask questions about computational chemistry, simulation parameters, literature, or generate code for your research.
                </p>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-2xl rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-chem-userBubble text-chem-text"
                    : "bg-chem-aiBubble text-chem-text"
                }`}
              >
                {msg.role === "assistant" ? (
                  <div className="prose prose-sm prose-invert max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-chem-aiBubble rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-chem-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
                  <div className="w-2 h-2 bg-chem-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
                  <div className="w-2 h-2 bg-chem-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-chem-border bg-chem-panel">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask ChemMind a question..."
              className="flex-1 bg-chem-dark border border-chem-border rounded-xl px-4 py-3 text-sm text-chem-text placeholder-chem-text-muted focus:outline-none focus:border-chem-primary"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-chem-primary hover:bg-chem-primaryHover disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-medium text-sm transition-colors"
            >
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
