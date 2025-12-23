#!/bin/bash

# Replace __API_URL__ placeholder with actual Railway URL from environment variable
sed -i "s|__API_URL__|${RAILWAY_API_URL}|g" env-config.js
