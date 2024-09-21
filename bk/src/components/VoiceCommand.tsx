// components/VoiceCommand.tsx
import React, { useEffect } from 'react';

const VoiceCommand: React.FC = () => {
  useEffect(() => {
    const recognition = new (window.SpeechRecognition ||
      (window as any).webkitSpeechRecognition)();
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0][0].transcript;
      // Process the transcript with your LLM
    };
    recognition.start();
  }, []);

  return null;
};

export default VoiceCommand;
