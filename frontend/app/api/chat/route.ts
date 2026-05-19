import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const { question, material_system, simulation_software } = await request.json();

    if (!question) {
      return NextResponse.json(
        { error: "Question is required" },
        { status: 400 }
      );
    }

    const backendUrl = process.env.CHEMMIND_BACKEND_URL || "http://localhost:8000";

    const response = await fetch(`${backendUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        material_system,
        simulation_software,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      response: data.response,
      agent_used: data.agent_used,
      intent: data.intent,
    });
  } catch (error) {
    console.error("ChemMind API error:", error);
    return NextResponse.json(
      { response: "Error: Could not connect to ChemMind backend. Ensure the Python server is running." },
      { status: 500 }
    );
  }
}
