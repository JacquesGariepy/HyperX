// components/GlitchText.tsx
import React from 'react';
import styled, { keyframes } from 'styled-components';

const glitch = keyframes`
  0% { clip-path: inset(0 0 85% 0); }
  10% { clip-path: inset(15% 0 0 0); }
  20% { clip-path: inset(0 0 70% 0); }
  30% { clip-path: inset(50% 0 0 0); }
  40% { clip-path: inset(0 0 30% 0); }
  50% { clip-path: inset(20% 0 0 0); }
  60% { clip-path: inset(0 0 80% 0); }
  70% { clip-path: inset(25% 0 0 0); }
  80% { clip-path: inset(0 0 40% 0); }
  90% { clip-path: inset(10% 0 0 0); }
  100% { clip-path: inset(0 0 85% 0); }
`;

const GlitchText = styled.h1`
  position: relative;
  color: ${(props) => props.theme.colors.primary};
  font-size: 4em;
  animation: ${glitch} 2s infinite;
`;

const GlitchComponent: React.FC = () => {
  return <GlitchText>Welcome to the Future</GlitchText>;
};

export default GlitchComponent;
