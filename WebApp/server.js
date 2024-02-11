const express = require('express');
const path = require('path');
const fetch = require('node-fetch');

const app = express();
const port = process.env.PORT || 8080;

// Serve static files (HTML, CSS, JS, images, etc.)
app.use(express.static(path.join(__dirname, '')));

// Forward POST requests to the Flask server
app.post('/', async (req, res) => {
  try {
    const flaskServerUrl = 'http://localhost:8080'; // Adjust the URL if necessary

    const flaskResponse = await fetch(`${flaskServerUrl}/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const flaskData = await flaskResponse.json();

    // Send the Flask response along with the Plotly data
    res.json({
      prediction_result: flaskData.prediction_result,
    });
  } catch (error) {
    console.error('Error forwarding request to Flask server:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Handle any other routes by serving the index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'site.html'));
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
