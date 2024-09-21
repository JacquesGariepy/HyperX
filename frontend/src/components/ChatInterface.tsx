import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Box, Input, Button, VStack, Text, useToast, Flex } from '@chakra-ui/react';

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
    <Box maxWidth="600px" width="100%" p={4} bg="rgba(255, 255, 255, 0.1)" borderRadius="lg" backdropFilter="blur(10px)">
      <VStack spacing={4} align="stretch" height="400px" overflowY="auto" mb={4}>
        {messages.map(message => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Flex justify={message.sender === 'user' ? 'flex-end' : 'flex-start'}>
              <Box
                bg={message.sender === 'user' ? 'blue.500' : 'whiteAlpha.300'}
                color={message.sender === 'user' ? 'white' : 'white'}
                p={3}
                borderRadius="lg"
                maxWidth="80%"
              >
                <Text>{message.text}</Text>
              </Box>
            </Flex>
          </motion.div>
        ))}
      </VStack>
      <Flex>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          mr={2}
          bg="whiteAlpha.200"
          color="white"
          _placeholder={{ color: 'whiteAlpha.600' }}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
        />
        <Button onClick={handleSend} colorScheme="blue">
          Send
        </Button>
      </Flex>
    </Box>
  );
};

export default ChatInterface;