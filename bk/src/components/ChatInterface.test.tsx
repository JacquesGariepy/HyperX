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
