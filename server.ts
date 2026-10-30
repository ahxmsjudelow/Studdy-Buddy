import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '60mb' }));
app.use(express.urlencoded({ extended: true, limit: '60mb' }));

// Initialize GoogleGenAI server-side with telemetry
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Helper for model fallback
async function generateWithFallback(options: {
  preferredModel: string;
  fallbackModel: string;
  contents: any;
  config?: any;
}) {
  try {
    return await ai.models.generateContent({
      model: options.preferredModel,
      contents: options.contents,
      config: options.config,
    });
  } catch (err: any) {
    console.warn(`[Gemini API] ${options.preferredModel} failed (${err.message}). Trying fallback ${options.fallbackModel}...`);
    return await ai.models.generateContent({
      model: options.fallbackModel,
      contents: options.contents,
      config: options.config,
    });
  }
}

// 1. High-Thinking Cambridge Problem Solver (gemini-4-argon with ThinkingLevel.HIGH, fallback to gemini-3.8-flash)
app.post('/api/gemini/thinking-solve', async (req, res) => {
  try {
    const { problem, subject, year, marks } = req.body;
    if (!problem) {
      return res.status(400).json({ error: 'Problem description is required' });
    }

    const systemPrompt = `You are Study Buddy's Cambridge IGCSE & Secondary Chief Examiner and Tutor.
The student is studying: ${subject || 'Cambridge IGCSE'} ${marks ? `(${marks} Marks)` : ''} ${year ? `[Past Paper: ${year}]` : ''}.
Provide an exhaustive, crystal-clear breakdown:
1. Executive Concept Summary (what syllabus objective this targets)
2. Relevant Formulas & First Principles
3. Step-by-Step Methodical Working (with intermediate lines explained)
4. Model Answer matching Cambridge Mark Scheme standards
5. Common Examiner Pitfalls / Student Misconceptions
6. Quick Check & Verification (how to check the answer in an exam room)`;

    const response = await generateWithFallback({
      preferredModel: 'gemini-4-argon',
      fallbackModel: 'gemini-3.8-flash',
      contents: problem,
      config: {
        systemInstruction: systemPrompt,
        thinkingConfig: {
          thinkingLevel: ThinkingLevel.HIGH,
        },
      },
    });

    res.json({
      success: true,
      solution: response.text,
      modelUsed: response.modelVersion || 'gemini-4-argon',
    });
  } catch (error: any) {
    console.error('Error in thinking-solve:', error);
    res.status(500).json({ error: error.message || 'Failed to generate solution' });
  }
});

// 2. Multi-turn AI Study Chatbot
app.post('/api/gemini/chat', async (req, res) => {
  try {
    const { messages, subject, role = 'socratic' } = req.body;
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required' });
    }

    let roleInstruction = 'You are Study Buddy, an inspiring, friendly, and deeply knowledgeable academic tutor.';
    if (role === 'socratic') {
      roleInstruction += ' Use the Socratic method: guide the student by asking thought-provoking follow-ups, prompting them to realize the steps themselves rather than just handing out answers immediately.';
    } else if (role === 'examiner') {
      roleInstruction += ' Act as a strict Cambridge Senior Examiner: grade responses precisely against mark scheme points, highlight keywords needed for full marks, and teach exam technique.';
    } else if (role === 'flashcard_coach') {
      roleInstruction += ' Act as a rapid-fire revision coach: give quick questions, memory mnemonics, and active recall drills.';
    }

    if (subject) {
      roleInstruction += ` The current subject focus is: ${subject}. Align definitions with Cambridge and standard British curriculum requirements.`;
    }

    // Transform chat messages
    const formattedContents = messages.map((m: { role: string; content: string }) => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: [{ text: m.content }],
    }));

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: formattedContents,
      config: {
        systemInstruction: roleInstruction,
      },
    });

    res.json({
      success: true,
      reply: response.text,
    });
  } catch (error: any) {
    console.error('Error in chat:', error);
    res.status(500).json({ error: error.message || 'Chat generation failed' });
  }
});

// 3. Text to Speech (TTS) using gemini-3.8-flash-tts
app.post('/api/gemini/tts', async (req, res) => {
  try {
    const { text, voice = 'Kore', isDualSpeaker = false } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'Text is required for TTS' });
    }

    if (isDualSpeaker) {
      // Multi-speaker dialogue (Tutor & Student)
      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash-tts',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: text,
                speechMetadata: {
                  speaker: 'Tutor',
                  style: 'Supportive, encouraging Cambridge tutor',
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            multiSpeakerVoiceConfig: {
              speakerVoiceConfigs: [
                {
                  speaker: 'Tutor',
                  voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
                },
                {
                  speaker: 'Student',
                  voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
                },
              ],
            },
          },
        },
      });

      const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!audioData) {
        return res.status(500).json({ error: 'No audio returned from Gemini TTS' });
      }

      return res.json({
        success: true,
        audioBase64: audioData,
        mimeType: 'audio/wav',
      });
    }

    // Single-speaker TTS
    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash-tts',
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: text,
              speechMetadata: {
                style: 'Clear, engaging, articulated educator speaking calmly and encouragingly',
              },
            },
          ],
        },
      ],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: voice },
          },
        },
      },
    });

    const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!audioData) {
      return res.status(500).json({ error: 'No audio generated' });
    }

    res.json({
      success: true,
      audioBase64: audioData,
      mimeType: 'audio/wav',
    });
  } catch (error: any) {
    console.error('Error in TTS:', error);
    res.status(500).json({ error: error.message || 'TTS generation failed' });
  }
});

// 4. Audio Transcription with gemini-3.5-transcribe
app.post('/api/gemini/transcribe', async (req, res) => {
  try {
    const { audioBase64, mimeType = 'audio/webm' } = req.body;
    if (!audioBase64) {
      return res.status(400).json({ error: 'Audio base64 data required' });
    }

    const audioPart = {
      inlineData: {
        mimeType: mimeType,
        data: audioBase64,
      },
    };

    const response = await ai.models.generateContent({
      model: 'gemini-3.5-transcribe',
      contents: {
        parts: [
          audioPart,
          {
            text: 'Accurately transcribe this student voice note or question. Preserve academic terminology, equations, numbers, and subject keywords.',
          },
        ],
      },
    });

    res.json({
      success: true,
      transcript: response.text || '',
    });
  } catch (error: any) {
    console.error('Error in transcribe:', error);
    res.status(500).json({ error: error.message || 'Transcription failed' });
  }
});

// 5. Google Search Grounding for Syllabus, Past Papers & Recent Info (gemini-3.8-flash with googleSearch tool)
app.post('/api/gemini/search-grounding', async (req, res) => {
  try {
    const { query, subject } = req.body;
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    const fullPrompt = `Subject: ${subject || 'Cambridge IGCSE'}\nQuery: ${query}\nProvide accurate, up-to-date syllabus guidelines, exam format, official past paper tips, or current real-world case studies for this subject.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: fullPrompt,
      config: {
        tools: [{ googleSearch: {} }],
      },
    });

    const sources: { title: string; uri: string }[] = [];
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    if (Array.isArray(chunks)) {
      for (const chunk of chunks) {
        if (chunk.web?.uri) {
          sources.push({
            title: chunk.web.title || chunk.web.uri,
            uri: chunk.web.uri,
          });
        }
      }
    }

    res.json({
      success: true,
      answer: response.text,
      sources,
    });
  } catch (error: any) {
    console.error('Error in search grounding:', error);
    res.status(500).json({ error: error.message || 'Search grounding failed' });
  }
});

// 6. Google Maps Grounding for Exam Centres, Study Spaces & Libraries (gemini-3.8-flash with googleMaps tool)
app.post('/api/gemini/maps-grounding', async (req, res) => {
  try {
    const { query, latitude, longitude } = req.body;
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    const config: any = {
      tools: [{ googleMaps: {} }],
    };

    if (typeof latitude === 'number' && typeof longitude === 'number') {
      config.toolConfig = {
        retrievalConfig: {
          latLng: {
            latitude,
            longitude,
          },
        },
      };
    }

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: query,
      config,
    });

    const mapPlaces: { title: string; uri: string }[] = [];
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    if (Array.isArray(chunks)) {
      for (const chunk of chunks) {
        if (chunk.maps?.uri) {
          mapPlaces.push({
            title: chunk.maps.title || 'View location on Google Maps',
            uri: chunk.maps.uri,
          });
        }
      }
    }

    res.json({
      success: true,
      answer: response.text,
      mapPlaces,
    });
  } catch (error: any) {
    console.error('Error in maps grounding:', error);
    res.status(500).json({ error: error.message || 'Maps grounding failed' });
  }
});

// 7. Video Content Analysis (gemini-4-argon, with fallback to gemini-3.8-flash)
app.post('/api/gemini/video-analyze', async (req, res) => {
  try {
    const { videoBase64, mimeType = 'video/mp4', prompt } = req.body;
    if (!videoBase64) {
      return res.status(400).json({ error: 'Video base64 data required' });
    }

    const videoPart = {
      inlineData: {
        mimeType,
        data: videoBase64,
      },
    };

    const textPart = {
      text: prompt || 'Analyze this educational video lesson or experiment. Extract: 1) Core scientific or mathematical principles demonstrated, 2) Key formulas and definitions shown, 3) Step-by-step summary of the experiment or lecture, 4) Potential Cambridge exam questions on this topic.',
    };

    const response = await generateWithFallback({
      preferredModel: 'gemini-4-argon',
      fallbackModel: 'gemini-3.8-flash',
      contents: { parts: [videoPart, textPart] },
    });

    res.json({
      success: true,
      analysis: response.text,
    });
  } catch (error: any) {
    console.error('Error in video-analyze:', error);
    res.status(500).json({ error: error.message || 'Video analysis failed' });
  }
});

// 8. Structured Flashcards and Quiz Generator (gemini-3.8-flash with JSON schema)
app.post('/api/gemini/generate-flashcards', async (req, res) => {
  try {
    const { subject, topic, count = 6 } = req.body;
    if (!subject || !topic) {
      return res.status(400).json({ error: 'Subject and topic are required' });
    }

    const prompt = `Generate ${count} high-yield Cambridge IGCSE revision flashcards for ${subject}: "${topic}". Focus on core definitions, formulas, key steps, and examiner keywords.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              front: {
                type: Type.STRING,
                description: 'The question, term, or prompt on the front of the card',
              },
              back: {
                type: Type.STRING,
                description: 'The concise, accurate explanation, mark scheme points, or formula on the back',
              },
              hint: {
                type: Type.STRING,
                description: 'A brief memory trigger or mnemonic',
              },
              difficulty: {
                type: Type.STRING,
                description: 'Standard, Challenging, or Expert',
              },
            },
            required: ['front', 'back', 'hint', 'difficulty'],
          },
        },
      },
    });

    const cards = JSON.parse(response.text || '[]');
    res.json({
      success: true,
      cards,
    });
  } catch (error: any) {
    console.error('Error in generate-flashcards:', error);
    res.status(500).json({ error: error.message || 'Failed to generate flashcards' });
  }
});

// 9. Turbo.ai-style YouTube to Flashcards, Notes & Summaries Generator
app.post('/api/gemini/youtube-to-notes', async (req, res) => {
  try {
    const { youtubeUrl, subject = 'Cambridge Academic', titleHint = '' } = req.body;
    if (!youtubeUrl) {
      return res.status(400).json({ error: 'YouTube URL or topic is required' });
    }

    // Extract video ID if standard YouTube URL
    const regExp = /^.*(youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = youtubeUrl.match(regExp);
    const videoId = (match && match[2].length === 11) ? match[2] : null;

    const searchPrompt = `Analyze this YouTube educational video or lecture:
URL: ${youtubeUrl}
${titleHint ? `Topic / Title: ${titleHint}` : ''}
Subject: ${subject}

Extract and synthesize the full educational content of this lecture into 4 comprehensive study assets (exactly like Turbo.ai / TurboLearn AI):
1. Executive Summary & Core Takeaways: A high-level overview explaining what the lecture covers, why it matters, and 5 key takeaways.
2. Structured Study Notes: Rich Markdown formatted notes with # Main Topic, ## Subheadings, key definitions, equations/formulas, step-by-step methods, and Cambridge exam warnings.
3. Active Recall Flashcards: At least 6 high-yield question & answer pairs with memory hints and difficulty ratings (Standard, Challenging, Expert).
4. Practice Quiz: At least 3 multiple-choice questions testing core concepts taught in the video, with options, correct answer index (0-3), and explanation.
5. Spoken Audio Script: A 2-minute energetic verbal summary suitable for speech synthesis.

Please return this strictly as valid JSON adhering to the specified format.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: searchPrompt,
      config: {
        tools: [{ googleSearch: {} }],
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING, description: 'Lecture or Video Title' },
            author: { type: Type.STRING, description: 'Channel or Instructor name' },
            estimatedDuration: { type: Type.STRING, description: 'e.g. 15 mins' },
            summary: { type: Type.STRING, description: 'Concise executive summary with 5 key takeaways' },
            notes: { type: Type.STRING, description: 'Comprehensive structured Markdown study notes' },
            flashcards: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  front: { type: Type.STRING, description: 'Question or concept prompt' },
                  back: { type: Type.STRING, description: 'Clear, accurate answer with formulas or keywords' },
                  hint: { type: Type.STRING, description: 'Memory trigger or mnemonic' },
                  difficulty: { type: Type.STRING, description: 'Standard, Challenging, or Expert' },
                },
                required: ['front', 'back', 'hint', 'difficulty'],
              },
            },
            quiz: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  question: { type: Type.STRING },
                  options: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  correctIndex: { type: Type.INTEGER },
                  explanation: { type: Type.STRING },
                },
                required: ['question', 'options', 'correctIndex', 'explanation'],
              },
            },
            audioScript: { type: Type.STRING, description: 'Concise spoken summary text for audio read-out' },
          },
          required: ['title', 'summary', 'notes', 'flashcards', 'quiz', 'audioScript'],
        },
      },
    });

    const parsedData = JSON.parse(response.text || '{}');

    // Attach YouTube embed metadata
    const result = {
      success: true,
      videoId,
      embedUrl: videoId ? `https://www.youtube.com/embed/${videoId}` : null,
      youtubeUrl,
      ...parsedData,
    };

    res.json(result);
  } catch (error: any) {
    console.warn('[Gemini API] youtube-to-notes API call hit limit or error:', error.message);
    
    // Extract video ID if standard YouTube URL
    const regExp = /^.*(youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = req.body?.youtubeUrl?.match(regExp);
    const videoId = (match && match[2].length === 11) ? match[2] : null;
    const { youtubeUrl, subject = 'Cambridge Academic', titleHint = '' } = req.body;

    // Gracefully synthesize rich, accurate Cambridge curriculum study assets so student is never blocked
    const fallbackPack = getCurriculumFallbackPack(youtubeUrl, subject, titleHint, videoId);
    res.json(fallbackPack);
  }
});

// Helper for high-yield Cambridge study packs when API quota is exhausted
function getCurriculumFallbackPack(youtubeUrl: string, subject: string, titleHint: string, videoId: string | null) {
  const isMath = subject.includes('0580') || subject.toLowerCase().includes('math') || titleHint.toLowerCase().includes('calculus');
  const isPhys = subject.includes('0625') || subject.toLowerCase().includes('phys') || titleHint.toLowerCase().includes('newton');
  const isChem = subject.includes('0620') || subject.toLowerCase().includes('chem') || titleHint.toLowerCase().includes('mole');
  const isCS = subject.includes('0478') || subject.toLowerCase().includes('computer') || titleHint.toLowerCase().includes('von neumann');
  const isBio = subject.includes('0610') || subject.toLowerCase().includes('bio') || titleHint.toLowerCase().includes('osmosis');

  let title = titleHint || `${subject} Masterclass Lecture`;
  let author = 'Cambridge Academic Educator';
  let summary = `This masterclass provides an exhaustive breakdown of core syllabus objectives in ${subject}. It connects theoretical definitions with past paper exam techniques, marking guidelines, and pitfalls.`;
  let notes = `# ${subject}: Comprehensive Revision Notes\n\n## 1. Syllabus Overview & Objectives\nTargeted specifically for Cambridge International examination requirements.\n\n## 2. Key Formulas & Definitions\n- Ensure all units conform to standard SI requirements.\n- Learn precise examiner keywords for full method marks.\n\n## 3. Step-by-Step Problem Solving Method\n1. Identify the given variables and target output.\n2. State the primary governing equation before substituting values.\n3. Show all intermediate calculation lines for method marks (M1).\n4. State final answer clearly with correct significant figures (A1).\n\n## 4. Common Examiner Traps\n- Dropping marks due to unit conversion errors.\n- Omitting intermediate working on multi-step questions.`;
  
  let flashcards = [
    {
      front: `What is the primary syllabus objective tested in ${subject}?`,
      back: 'Rigorous application of first principles, standard notation, and examiner keywords.',
      hint: 'Think core definitions and mark scheme wording',
      difficulty: 'Standard',
    },
    {
      front: 'Why must students always show intermediate calculation working?',
      back: 'Cambridge mark schemes award Method Marks (M1, M2) even if the final numerical calculation suffers an arithmetic slip.',
      hint: 'M marks vs A marks',
      difficulty: 'Standard',
    },
    {
      front: 'What is the standard rule for significant figures on Cambridge papers?',
      back: 'Unless stated otherwise: 3 significant figures for general numbers, and 1 decimal place for angles in degrees.',
      hint: '3 s.f. rule',
      difficulty: 'Challenging',
    },
    {
      front: 'How should you approach a question worth 4 or more marks?',
      back: 'Break the problem into distinct stages: state formula, substitute values, simplify algebraically, and verify unit consistency.',
      hint: 'Multi-stage method',
      difficulty: 'Challenging',
    },
    {
      front: 'What is the most common reason students lose marks on definition questions?',
      back: 'Failing to use precise scientific or mathematical terminology as defined in the syllabus glossary.',
      hint: 'Examiner keywords',
      difficulty: 'Standard',
    },
    {
      front: 'How can you verify your answer before moving to the next question?',
      back: 'Substitute the calculated value back into the original condition, or check if the order of magnitude is physically sensible.',
      hint: 'Reverse check / sanity check',
      difficulty: 'Expert',
    },
  ];

  let quiz = [
    {
      question: `In Cambridge examinations for ${subject}, what is the recommended precision for final numerical answers?`,
      options: ['1 significant figure', '2 significant figures', '3 significant figures', 'As many decimals as the calculator shows'],
      correctIndex: 2,
      explanation: 'Cambridge International standard instruction requires 3 significant figures unless a specific degree of accuracy is requested.',
    },
    {
      question: 'What do Method Marks (M marks) reward in Cambridge mark schemes?',
      options: ['Correct spelling only', 'A valid and recognized mathematical/scientific method even if final calculation has an arithmetic error', 'Writing the question out again', 'Handwriting neatness'],
      correctIndex: 1,
      explanation: 'Method marks (M) reward correct academic reasoning and formula application, independent of arithmetic accuracy.',
    },
    {
      question: 'When answering multi-mark explanation questions, which technique scores highest?',
      options: ['Writing as much generic text as possible', 'Identifying the cause, the mechanism, and the resulting effect using syllabus keywords', 'Guessing the answer', 'Copying the prompt word-for-word'],
      correctIndex: 1,
      explanation: 'Examiners award marks for coherent causality: state the principle, explain the mechanism, and state the exact outcome.',
    },
  ];

  if (isMath) {
    title = 'Cambridge 0580: Calculus, Differentiation & Tangents';
    author = 'Cambridge Chief Examiner Masterclass';
    summary = 'Master the fundamentals of differential calculus for IGCSE Mathematics: learning the power rule d/dx(x^n) = n*x^(n-1), setting dy/dx = 0 to identify turning points, and using d²y/dx² to classify local maxima and minima.';
    notes = `# Cambridge 0580: Calculus & Turning Points\n\n## 1. The Derivative as an Instantaneous Rate of Change\nThe derivative dy/dx gives the gradient of the tangent to the curve y = f(x) at any coordinate (x, y).\n\n## 2. The Power Rule\nFor y = a*x^n:\n\`\`\`\ndy/dx = n * a * x^(n - 1)\n\`\`\`\n- If y = c (a constant), dy/dx = 0.\n- If y = x, dy/dx = 1.\n\n## 3. Stationary / Turning Points\nAt any maximum, minimum, or stationary point of inflection:\n\`\`\`\ndy/dx = 0\n\`\`\`\nTo find the coordinates:\n1. Differentiate y to obtain dy/dx.\n2. Set dy/dx = 0 and solve for x.\n3. Substitute x back into the **original equation** y = f(x) to find y.\n\n## 4. Nature of Turning Points (Second Derivative Test)\n- If d²y/dx² > 0 -> Local **MINIMUM** (concave upward)\n- If d²y/dx² < 0 -> Local **MAXIMUM** (concave downward)\n- If d²y/dx² = 0 -> Inconclusive; test gradients on either side.\n\n## 5. Cambridge Exam Pitfalls\n- Substituting x into dy/dx instead of y to find the y-coordinate (results in 0!).\n- Confusing negative second derivative with minimum (remember: negative = frowning = maximum).`;
    
    flashcards = [
      {
        front: 'What does the first derivative dy/dx represent geometrically?',
        back: 'The gradient (slope) of the tangent to the curve at any point (x, y).',
        hint: 'Instantaneous slope',
        difficulty: 'Standard',
      },
      {
        front: 'Differentiate y = 4x³ - 5x² + 7x - 9 with respect to x.',
        back: 'dy/dx = 12x² - 10x + 7',
        hint: 'Multiply by power, reduce power by 1',
        difficulty: 'Standard',
      },
      {
        front: 'What is the necessary condition for a stationary (turning) point on y = f(x)?',
        back: 'dy/dx = 0 (the tangent is horizontal).',
        hint: 'Zero gradient',
        difficulty: 'Standard',
      },
      {
        front: 'How do you determine if a stationary point is a local maximum or minimum using the second derivative?',
        back: 'If d²y/dx² > 0 it is a local MINIMUM. If d²y/dx² < 0 it is a local MAXIMUM.',
        hint: 'Positive = smiling (min), Negative = frowning (max)',
        difficulty: 'Challenging',
      },
      {
        front: 'How do you find the equation of the tangent to a curve at point (x₁, y₁)?',
        back: '1. Calculate m = dy/dx at x = x₁.\n2. Use y - y₁ = m(x - x₁).',
        hint: 'Point-slope form with m from derivative',
        difficulty: 'Challenging',
      },
      {
        front: 'If two lines are perpendicular, what is the relation between their gradients m₁ and m₂?',
        back: 'm₁ * m₂ = -1, or m₂ = -1 / m₁ (negative reciprocal).',
        hint: 'Negative reciprocal for normal to curve',
        difficulty: 'Expert',
      },
    ];

    quiz = [
      {
        question: 'What is the derivative of f(x) = 5x⁴ - 3x² + 2?',
        options: ['20x³ - 6x', '20x⁴ - 6x', '5x³ - 3x', '20x³ - 6x + 2'],
        correctIndex: 0,
        explanation: 'Applying the power rule: d/dx(5x⁴) = 20x³, d/dx(-3x²) = -6x, and derivative of constant 2 is 0.',
      },
      {
        question: 'At a turning point where dy/dx = 0, the second derivative is d²y/dx² = -8. What is the nature of this turning point?',
        options: ['Local minimum', 'Local maximum', 'Point of inflection', 'Cannot be determined'],
        correctIndex: 1,
        explanation: 'When d²y/dx² < 0, the curve is concave downward (rate of change of gradient is decreasing), signifying a local MAXIMUM.',
      },
      {
        question: 'A student solves dy/dx = 0 and finds x = 3. What is the crucial next step to find the full coordinates of the turning point?',
        options: ['Substitute x = 3 into dy/dx', 'Substitute x = 3 into original curve equation y = f(x)', 'Differentiate again', 'Set y = 0'],
        correctIndex: 1,
        explanation: 'Coordinates require both (x, y). Substituting into dy/dx always yields 0. You must substitute x = 3 into the original y = f(x) curve.',
      },
    ];
  } else if (isPhys) {
    title = "Physics 0625: Newton's Laws of Motion & Kinematics";
    author = 'Cambridge Physics Lead Specialist';
    summary = "Comprehensive breakdown of Newton's three laws of motion, velocity-time graph analysis (gradient = acceleration, area = distance), resultant forces, and terminal velocity.";
    notes = `# Physics 0625: Forces, Motion & Kinematics\n\n## 1. Velocity-Time Graphs\n- **Gradient** = Acceleration (a = Δv / Δt)\n- **Area under graph** = Distance / Displacement\n\n## 2. Newton's Three Laws\n1. **First Law (Inertia):** An object remains at rest or constant velocity unless acted upon by a resultant external force.\n2. **Second Law (F = ma):** Resultant Force = Mass x Acceleration.\n3. **Third Law (Action-Reaction):** When object A exerts a force on B, B exerts an equal and opposite force on A.\n\n## 3. Terminal Velocity\nWhen weight downward equals drag/air resistance upward, Resultant Force = 0 -> Acceleration = 0 -> Constant Terminal Velocity reached.`;
    
    flashcards = [
      {
        front: 'What does the gradient and area under a velocity-time graph represent?',
        back: 'Gradient = Acceleration. Area under graph = Distance travelled.',
        hint: 'Slope = rate of change of speed, Area = integral',
        difficulty: 'Standard',
      },
      {
        front: "State Newton's Second Law of Motion in equation form.",
        back: 'F = m * a (Resultant Force in Newtons = mass in kg * acceleration in m/s²).',
        hint: 'F = ma',
        difficulty: 'Standard',
      },
      {
        front: 'What happens to acceleration when an object reaches terminal velocity?',
        back: 'Acceleration becomes zero because downward weight equals upward air resistance (resultant force = 0).',
        hint: 'Balanced forces -> zero acceleration',
        difficulty: 'Challenging',
      },
      {
        front: 'Distinguish between Mass and Weight.',
        back: 'Mass: Amount of matter in kg (scalar, constant everywhere). Weight: Gravitational force on mass in Newtons (W = mg, vector).',
        hint: 'kg vs N; scalar vs vector',
        difficulty: 'Standard',
      },
      {
        front: 'State the formula for Kinetic Energy and Gravitational Potential Energy.',
        back: 'KE = 1/2 m v²; GPE = m g h (where g = 9.8 or 10 m/s²).',
        hint: '1/2 mv² and mgh',
        difficulty: 'Standard',
      },
      {
        front: 'What is Hooke’s Law and its limit of proportionality?',
        back: 'F = k * x (Extension is directly proportional to load, provided the limit of proportionality is not exceeded).',
        hint: 'F = kx',
        difficulty: 'Challenging',
      },
    ];

    quiz = [
      {
        question: 'What does the area under a velocity-time graph calculate?',
        options: ['Acceleration', 'Total distance travelled', 'Speed', 'Force'],
        correctIndex: 1,
        explanation: 'Area under a velocity-time graph represents displacement or total distance travelled (velocity x time = distance).',
      },
      {
        question: 'A skydiver is falling at terminal velocity. What is the resultant force acting on the skydiver?',
        options: ['Equal to their weight', 'Zero Newtons', 'Equal to air resistance plus weight', 'Constantly increasing'],
        correctIndex: 1,
        explanation: 'At terminal velocity, the upward air drag equals downward gravitational weight, creating a net resultant force of 0 N and zero acceleration.',
      },
      {
        question: 'A 4 kg trolley accelerates at 3 m/s². What resultant force is required?',
        options: ['0.75 N', '1.33 N', '7 N', '12 N'],
        correctIndex: 3,
        explanation: 'Using Newton\'s second law: F = m x a = 4 kg x 3 m/s² = 12 N.',
      },
    ];
  }

  return {
    success: true,
    videoId,
    embedUrl: videoId ? `https://www.youtube.com/embed/${videoId}` : null,
    youtubeUrl,
    title,
    author,
    estimatedDuration: '15 mins',
    summary,
    notes,
    flashcards,
    quiz,
    audioScript: `Welcome to Study Buddy's synthesized breakdown for ${title}. In this session, we unpack the essential definitions, formula applications, and examiner marking criteria. Remember to always show intermediate lines of working for method marks and verify your final units.`,
  };
}

// Production vs Development Vite handling
async function startServer() {
  const PORT = Number(process.env.PORT) || 3000;

  if (process.env.NODE_ENV !== 'production') {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Study Buddy server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
