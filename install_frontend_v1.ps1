# frontend_setup.ps1

# Function to check if a command exists
function Test-Command($command) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { return $true } }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Check for required tools
$requiredCommands = @("node", "npm")
foreach ($cmd in $requiredCommands) {
    if (-not (Test-Command $cmd)) {
        Write-Error "$cmd is not installed or not in PATH. Please install it and try again."
        exit 1
    }
}

# Create project directory
$projectName = "frontend"
New-Item -ItemType Directory -Force -Path $projectName
Set-Location $projectName

# Initialize React app with TypeScript
npx create-react-app . --template typescript

# Install additional dependencies
npm install three @types/three framer-motion @chakra-ui/react @emotion/react @emotion/styled axios @testing-library/react @testing-library/jest-dom @testing-library/user-event

# Create component directory
New-Item -ItemType Directory -Force -Path "src/components"

# Create Background.tsx
@"
import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

const Background: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true });

    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const geometry = new THREE.IcosahedronGeometry(1, 1);
    const material = new THREE.MeshPhongMaterial({
      color: 0x00ffff,
      wireframe: true,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const light = new THREE.PointLight(0xffffff, 1, 100);
    light.position.set(0, 0, 10);
    scene.add(light);

    camera.position.z = 4;

    const animate = () => {
      requestAnimationFrame(animate);
      mesh.rotation.x += 0.01;
      mesh.rotation.y += 0.01;
      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={mountRef} style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }} />;
};

export default Background;
"@ | Out-File -FilePath "src/components/Background.tsx" -Encoding utf8

# Create ChatInterface.tsx
@"
import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Box, Input, Button, VStack, Text, useToast } from '@chakra-ui/react';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'ai';
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const toast = useToast();

  const handleSend = async () => {
    if (input.trim()) {
      setMessages([...messages, { id: Date.now(), text: input, sender: 'user' }]);
      setInput('');
      try {
        const response = await axios.post(`${process.env.REACT_APP_API_URL}/chat`, { content: input });
        setMessages(msgs => [...msgs, { id: Date.now(), text: response.data.response, sender: 'ai' }]);
      } catch (error) {
        console.error('Error sending message:', error);
        toast({
          title: 'Error',
          description: 'Could not send message. Please try again.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    }
  };

  return (
    <Box maxWidth="600px" margin="auto" p={4} bg="rgba(255, 255, 255, 0.1)" borderRadius="lg" backdropFilter="blur(10px)">
      <VStack spacing={4} align="stretch" height="400px" overflowY="auto" mb={4}>
        {messages.map(message => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Box
              bg={message.sender === 'user' ? 'blue.500' : 'whiteAlpha.300'}
              color={message.sender === 'user' ? 'white' : 'white'}
              p={2}
              borderRadius="lg"
              alignSelf={message.sender === 'user' ? 'flex-end' : 'flex-start'}
            >
              <Text>{message.text}</Text>
            </Box>
          </motion.div>
        ))}
      </VStack>
      <Box display="flex">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          mr={2}
          bg="whiteAlpha.200"
          color="white"
          _placeholder={{ color: 'whiteAlpha.600' }}
        />
        <Button onClick={handleSend} colorScheme="blue">Send</Button>
      </Box>
    </Box>
  );
};

export default ChatInterface;
"@ | Out-File -FilePath "src/components/ChatInterface.tsx" -Encoding utf8

# Modify App.tsx
@"
import React from 'react';
import { ChakraProvider, Box, VStack, Heading } from '@chakra-ui/react';
import Background from './components/Background';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <ChakraProvider>
      <Box minHeight="100vh" bg="gray.900" color="white">
        <Background />
        <VStack spacing={8} justify="center" minHeight="100vh">
          <Heading as="h1" size="2xl" textAlign="center" mb={8}>
            Hyperbolic LLM Chat
          </Heading>
          <ChatInterface />
        </VStack>
      </Box>
    </ChakraProvider>
  );
}

export default App;
"@ | Out-File -FilePath "src/App.tsx" -Encoding utf8

# Create test file for ChatInterface
@"
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import ChatInterface from './ChatInterface';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

test('renders chat interface', () => {
  render(<ChatInterface />);
  const inputElement = screen.getByPlaceholderText(/Type your message.../i);
  const sendButton = screen.getByText(/Send/i);
  expect(inputElement).toBeInTheDocument();
  expect(sendButton).toBeInTheDocument();
});

test('sends a message and receives a response', async () => {
  mockedAxios.post.mockResolvedValue({ data: { response: 'Hello, human!' } });

  render(<ChatInterface />);
  const inputElement = screen.getByPlaceholderText(/Type your message.../i) as HTMLInputElement;
  const sendButton = screen.getByText(/Send/i);

  fireEvent.change(inputElement, { target: { value: 'Hello, AI!' } });
  fireEvent.click(sendButton);

  expect(screen.getByText('Hello, AI!')).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByText('Hello, human!')).toBeInTheDocument();
  });
});
"@ | Out-File -FilePath "src/components/ChatInterface.test.tsx" -Encoding utf8

# Create .env file
@"
REACT_APP_API_URL=http://localhost:8000
"@ | Out-File -FilePath ".env" -Encoding utf8

# Modify package.json to add proxy
$packageJson = Get-Content .\package.json -Raw | ConvertFrom-Json
$packageJson | Add-Member -Type NoteProperty -Name "proxy" -Value "http://localhost:8000"
$packageJson | ConvertTo-Json -Depth 32 | Set-Content .\package.json

# Update .gitignore
@"
# dependencies
/node_modules
/.pnp
.pnp.js

# testing
/coverage

# production
/build

# misc
.DS_Store
.env.local
.env.development.local
.env.test.local
.env.production.local

npm-debug.log*
yarn-debug.log*
yarn-error.log*

# environment variables
.env
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# Initialize git repository
git init
git add .
git commit -m "Initial frontend setup with 3D background and chat interface"

# Final message
Write-Host "Frontend setup complete!" -ForegroundColor Green
Write-Host "To start the development server, run 'npm start' in this directory." -ForegroundColor Yellow
Write-Host "To run tests, use 'npm test'." -ForegroundColor Yellow