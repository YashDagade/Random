#!/bin/bash

# List of IP addresses to check (from your arp -a output)
IPS=(
    "10.197.5.110"
    "10.197.7.171"
    "10.197.35.11"
    "10.197.137.57"
    "10.197.152.163"
    "10.197.162.169"
    "10.197.219.2"
    "10.197.227.81"
    "10.197.231.129"
)

# Username for the Raspberry Pi
USERNAME="yash"

# Try pinging each IP address and SSH if ping is successful
for IP in "${IPS[@]}"; do
    echo "Pinging $IP..."
    if ping -c 1 "$IP" &> /dev/null; then
        echo "Ping successful! Attempting SSH to $IP"
        # Attempt SSH connection with a 2-second timeout
        ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no "$USERNAME@$IP"
        if [ $? -eq 0 ]; then
            echo "Successfully connected to Raspberry Pi at $IP"
            exit 0
        else
            echo "SSH failed for $IP"
        fi
    else
        echo "Ping failed for $IP, moving to next IP..."
    fi
done

echo "Could not connect to any Raspberry Pi device. Please check the network or try a different method."
