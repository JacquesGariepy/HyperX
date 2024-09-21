// HolographicButton.tsx
import React from 'react';
import styled from 'styled-components';

export const HolographicButton: React.FC = () => {
  return <Button>Interagir</Button>;
};

const Button = styled.button`
  background: linear-gradient(45deg, #00ffab, #00e4ff);
  border: none;
  padding: 15px 30px;
  border-radius: 10px;
  color: #000;
  font-size: 1.2rem;
  font-weight: bold;
  cursor: pointer;
  box-shadow: 0 0 10px #00ffab, 0 0 40px #00ffab, 0 0 80px #00e4ff;
  transition: box-shadow 0.3s ease-in-out;

  &:hover {
    box-shadow: 0 0 20px #00ffab, 0 0 50px #00e4ff, 0 0 100px #00ffab;
  }
`;
