// components/NeuralNetworkVisualizer.tsx
import React from 'react';
import { Canvas } from '@react-three/fiber';
import { TorusKnot } from '@react-three/drei';

const NeuralNetworkVisualizer: React.FC = () => {
  return (
    <Canvas style={{ position: 'absolute', top: 0, left: 0 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <TorusKnot args={[1, 0.4, 100, 16]} position={[0, 0, 0]}>
        <meshStandardMaterial attach="material" color="#ff00e6" wireframe />
      </TorusKnot>
    </Canvas>
  );
};

export default NeuralNetworkVisualizer;
