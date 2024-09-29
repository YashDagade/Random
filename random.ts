import React, { useState } from 'react';
import { View, Text, Button } from 'react-native';
import axios from 'axios';

const SpeedLimitApp = () => {
  const [speedLimit, setSpeedLimit] = useState(null);
  const [error, setError] = useState(null);

  const getSpeedLimit = async (lat, long) => {
    const apiKey = 'YOUR_HERE_API_KEY'; // Replace with your HERE API Key
    const url = `https://router.hereapi.com/v8/routes?transportMode=car&origin=${lat},${long}&destination=${lat},${long}&return=summary&apikey=${apiKey}`;

    try {
      const response = await axios.get(url);
      const speedLimitData = response.data.routes[0]?.sections[0]?.summary?.maxSpeed;

      if (speedLimitData) {
        setSpeedLimit(speedLimitData);
      } else {
        setError('No speed limit data available for this location.');
      }
    } catch (err) {
      setError('Error fetching speed limit data.');
      console.error(err);
    }
  };

  return (
    <View style={{ padding: 20 }}>
      <Button title="Get Speed Limit" onPress={() => getSpeedLimit(40.730610, -73.935242)} />
      
      {speedLimit && <Text>Speed Limit: {speedLimit} km/h</Text>}
      {error && <Text>Error: {error}</Text>}
    </View>
  );
};

export default SpeedLimitApp;
