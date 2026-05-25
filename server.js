import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function systemPrompt({ userName = "Usuario", assistantName = "Andrea" }) {
  return `
Eres ${assistantName}, una guía emocional empática y breve.
Hablas en español.
Tus respuestas deben ser cortas, cálidas y humanas.
Máximo 2 frases.
No hagas respuestas largas.
Si es de mañana, suena motivadora.
Si es de noche, suena tranquila y cercana.
Usuario: ${userName}.
`;
}

app.post("/chat", async (req, res) => {
  try {
    const {
      message,
      history = [],
      userName = "Usuario",
      assistantName = "Andrea",
    } = req.body;

    const input = [
      {
        role: "system",
        content: systemPrompt({ userName, assistantName }),
      },
      ...history.map((m) => ({
        role: m.role,
        content: m.content,
      })),
      {
        role: "user",
        content: message,
      },
    ];

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input,
    });

    const reply =
      response.output_text?.trim() || "Estoy aquí contigo. Cuéntame un poco más.";

    res.json({ reply });
  } catch (error) {
    console.error("Error en /chat:", error);
    res.status(500).json({
      reply: "Estoy aquí contigo. Cuéntame un poco más.",
    });
  }
});

app.listen(process.env.PORT || 3000, () => {
  console.log("Backend escuchando en puerto " +
    (process.env.PORT || 3000));
});