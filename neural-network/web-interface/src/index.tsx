import React from 'react';
import ReactDOM from 'react-dom/client';
import axios from 'axios';

// Simple React component
const NeuralNetworkApp: React.FC = () => {
  const [message, setMessage] = React.useState('Neural Network Web Interface loaded!');
  
  React.useEffect(() => {
    // Test API connection
    axios.get('/api/v1/health')
      .then(response => {
        setMessage(`Connected to API: ${response.data.status}`);
      })
      .catch(error => {
        setMessage(`API Error: ${error.message}`);
      });
  }, []);

  return React.createElement('div', {
    style: { 
      padding: '20px', 
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f0f0f0',
      minHeight: '100vh'
    }
  }, [
    React.createElement('h1', { key: 'title' }, '🤖 Neural Network Interface'),
    React.createElement('p', { key: 'message' }, message),
    React.createElement('div', {
      key: 'status',
      style: {
        backgroundColor: 'white',
        padding: '20px',
        borderRadius: '8px',
        marginTop: '20px'
      }
    }, [
      React.createElement('h2', { key: 'status-title' }, 'Services Status:'),
      React.createElement('ul', { key: 'status-list' }, [
        React.createElement('li', { key: 'go' }, '✅ Go Service: Running on port 8090'),
        React.createElement('li', { key: 'rust' }, '🦀 Rust Service: Running on port 8080'),
        React.createElement('li', { key: 'python' }, '🐍 Python Service: Running on port 8081')
      ])
    ])
  ]);
};

// Render the app
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(React.createElement(NeuralNetworkApp));