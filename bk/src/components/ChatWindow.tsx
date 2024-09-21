// ChatWindow.tsx
import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

export const ChatWindow: React.FC = () => {
  return (
    <Window>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <Message>Bienvenue dans l'interface AI. Posez votre question !</Message>
      </motion.div>
    </Window>
  );
};

const Window = styled.div`
  width: 600px;
  height: 400px;
  border: 1px solid #00ffab;
  padding: 20px;
  background: rgba(0, 0, 0, 0.5);
  box-shadow: 0 0 15px #00ffab;
  overflow-y: auto;
`;

const Message = styled.div`
  background: rgba(0, 255, 171, 0.2);
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 10px;
  color: #fff;
`;
