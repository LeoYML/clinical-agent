import React, { useState } from 'react';
import axios from 'axios';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import {
  Container, 
  TextField, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  CircularProgress, 
  Grid, 
  Typography, 
  Box,
  AppBar,
  Toolbar,
  Card,
  CardContent,
  Avatar
} from '@mui/material';
import FaceIcon from '@mui/icons-material/Face';
import ChatIcon from '@mui/icons-material/Chat';
import ExternalSources from './api/external';

const theme = createTheme({
  palette: {
    primary: {
      // Using a soft blue that's calming and often associated with healthcare
      main: '#0097A7',
    },
    secondary: {
      // Opting for a gentle green that complements the primary color and evokes feelings of wellness and safety
      main: '#48D1CC',
    },
    // Adding an error color that's softer than the default red, making error messages less stressful
    error: {
      main: '#E57373',
    },
    // Including background colors that are easy on the eyes for both light and dark modes
    background: {
      default: '#f4f4f4',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    // Ensuring text is accessible and easy to read; adjusting font sizes and weights appropriately
    fontSize: 14,
    h1: {
      fontWeight: 500,
      fontSize: '2.2rem', // Slightly larger for clear hierarchy
    },
    h2: {
      fontWeight: 500,
      fontSize: '1.8rem',
    },
    h3: {
      fontWeight: 500,
      fontSize: '1.5rem',
    },
    h4: {
      fontWeight: 500, // Adjusting weight for readability
      fontSize: '1.25rem',
    },
    body1: {
      fontWeight: 400,
      fontSize: '1rem',
    },
    body2: {
      fontWeight: 400,
      fontSize: '.875rem',
    },
    button: {
      fontWeight: 500,
      textTransform: 'none', // Removing capitalization for a more friendly button text
    },
  },
  // Enhancing accessibility with larger touch targets for interactive elements
  shape: {
    borderRadius: 8, // Softening edges for a friendlier interface
  },
  // Customizing component defaults to fit the health care context and ensure consistency
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          fontSize: '1rem',
          padding: '6px 16px', // Enhancing button readability and touch friendliness
        },
      },
    },
    // Other component overrides can go here
  },
});


export default function Home() {
  const [message, setMessage] = useState('');
  const [responses, setResponses] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!message.trim()) return;
    setIsLoading(true);

    const messagesHistory = responses.map(resp => ({ role: resp.role, content: resp.message }));
    messagesHistory.push({ role: "user", content: message });

    try {
      const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        {
          model: "gpt-3.5-turbo",
          messages: messagesHistory,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.NEXT_PUBLIC_OPENAI_API_KEY}`,
          },
        }
      );

      setResponses([...responses, 
        { role: 'user', message: message },
        { role: 'assistant', message: response.data.choices[0].message.content.trim() }
      ]);

      setMessage('');
    } catch (error) {
      alert('Error sending message. Please check the console for more details.');
      console.error('Error sending message: ', error);
    }
    setIsLoading(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">ClinicalAgent Chat</Typography>
        </Toolbar>
      </AppBar>
      <ExternalSources />
      <Container maxWidth="md">
        <List>
          {responses.map((entry, index) => (
            <ListItem key={index}>
              <Card variant="outlined" sx={{ width: '100%', bgcolor: entry.role === 'user' ? 'primary.light' : 'secondary.light' }}>
                <CardContent>
                  <Box display="flex" alignItems="center">
                    <Avatar sx={{ bgcolor: entry.role === 'user' ? 'primary.main' : 'secondary.main', mr: 2 }}>
                      {entry.role === 'user' ? <FaceIcon /> : <ChatIcon />}
                    </Avatar>
                    <ListItemText primary={`${entry.role === 'user' ? 'You' : 'ClinicalAgent'}: ${entry.message}`} />
                  </Box>
                </CardContent>
              </Card>
            </ListItem>
          ))}
        </List>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={9}>
            <TextField
              label="Type your message here..."
              variant="outlined"
              fullWidth
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              disabled={isLoading}
            />
          </Grid>
          <Grid item xs={12} sm={3}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleSendMessage} 
              disabled={isLoading}
              fullWidth
            >
              Send
            </Button>
          </Grid>
          {isLoading && (
            <Grid item xs={12}>
              <Box display="flex" justifyContent="center">
                <CircularProgress />
              </Box>
            </Grid>
          )}
        </Grid>
      </Container>
    </ThemeProvider>
  );
}
