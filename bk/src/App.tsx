import React from 'react';
import { ChakraProvider, Box, VStack, Heading, Text, Container } from '@chakra-ui/react';
import { motion, MotionProps } from 'framer-motion';
import Background from './components/Background';
import ChatInterface from './components/ChatInterface';

// Créer un type pour MotionBox qui combine les props de Box et Motion
type MotionBoxProps = Omit<React.ComponentProps<typeof Box>, keyof MotionProps> & MotionProps;

// Créer le composant MotionBox
const MotionBox: React.FC<MotionBoxProps> = motion(Box as any);

const App: React.FC = () => {
  return (
    <ChakraProvider>
      <Box minHeight="100vh" bg="gray.900" color="white" position="relative" overflow="hidden">
        <Background />
        <Container maxW="container.xl" centerContent>
          <VStack spacing={8} justify="center" minHeight="100vh">
            <MotionBox
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <Heading as="h1" size="2xl" textAlign="center" mb={2}>
                Hyperbolic LLM Chat
              </Heading>
              <Text fontSize="xl" textAlign="center" mb={8}>
                Experience the future of AI-powered conversations
              </Text>
            </MotionBox>
            <ChatInterface />
          </VStack>
        </Container>
      </Box>
    </ChakraProvider>
  );
}

export default App;