# HeronAI Document Classifier: Frontend Setup Guide

This document outlines the implementation plan for a React TypeScript frontend to interact with the HeronAI Document Classifier service. It provides setup instructions, architecture guidance, and best practices for building a modern, responsive user interface.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Architecture & Components](#architecture--components)
- [Key Features](#key-features)
- [API Integration](#api-integration)
- [State Management](#state-management)
- [Styling Approach](#styling-approach)
- [Deployment](#deployment)
- [Example Component Implementation](#example-component-implementation)

## Overview

The HeronAI frontend provides a user-friendly interface for uploading documents, viewing classification results, and managing batch processing jobs. It features a modern, responsive design with drag-and-drop file upload, real-time status updates, and rich visualizations of classification results.

## Prerequisites

- Node.js 18+ and npm/yarn
- Basic knowledge of React, TypeScript, and modern frontend tools
- Access to the HeronAI API (development or production endpoint)

## Project Setup

### 1. Create a New Project with Vite

```bash
# Using npm
npm create vite@latest heronai-frontend -- --template react-ts

# Using yarn
yarn create vite heronai-frontend --template react-ts

# Navigate to the project
cd heronai-frontend
```

### 2. Install Dependencies

```bash
# Using npm
npm install axios react-router-dom @tanstack/react-query @mantine/core @mantine/hooks @mantine/dropzone @mantine/notifications @emotion/react recharts

# Using yarn
yarn add axios react-router-dom @tanstack/react-query @mantine/core @mantine/hooks @mantine/dropzone @mantine/notifications @emotion/react recharts
```

### 3. Configure API Base URL

Create a `.env` file in the project root:

```
VITE_API_BASE_URL=http://localhost:8000
VITE_API_KEY=your_api_key_here
```

### 4. Set Up Development Server

```bash
# Using npm
npm run dev

# Using yarn
yarn dev
```

## Architecture & Components

The frontend follows a modular architecture with the following key components:

```
src/
├── api/                  # API integration layer
│   ├── client.ts         # Axios instance & interceptors
│   ├── documents.ts      # Document classification API
│   └── jobs.ts           # Async job status API
├── components/           # Reusable UI components
│   ├── FileUpload/       # Drag & drop file upload
│   ├── ClassificationResult/ # Result display
│   ├── BatchJobStatus/   # Async job tracking
│   ├── Navigation/       # App navigation
│   └── common/           # Buttons, cards, etc.
├── pages/                # Application pages
│   ├── Dashboard.tsx     # Main dashboard
│   ├── Upload.tsx        # File upload page
│   ├── Results.tsx       # Results display
│   ├── Jobs.tsx          # Job management
│   └── Settings.tsx      # API configuration
├── hooks/                # Custom React hooks
│   ├── useClassification.ts # Classification logic
│   ├── useJobPolling.ts     # Job status polling
│   └── useSettings.ts       # App settings
├── utils/                # Utility functions
│   ├── formatters.ts     # Date/number formatting
│   └── validators.ts     # Input validation
├── context/              # React context providers
│   ├── AuthContext.tsx   # API key management
│   └── ThemeContext.tsx  # Theme settings
├── types/                # TypeScript type definitions
│   ├── api.ts            # API response types
│   └── app.ts            # Application types
├── App.tsx               # Main application component
└── main.tsx              # Application entry point
```

## Key Features

1. **Interactive Upload Interface**

   - Drag-and-drop file upload with preview
   - Multi-file selection with batch upload support
   - File type validation and size restriction
   - Upload progress indication

2. **Classification Results Display**

   - Card-based layout showing each document's classification
   - Confidence score visualization (gauge charts)
   - Stage-by-stage confidence breakdown
   - Warning and error display

3. **Async Job Management**

   - Job queue visualization
   - Real-time status updates via polling
   - Completion notifications
   - Result download options

4. **Dashboard & Analytics**
   - Recent upload history
   - Classification distribution charts
   - Confidence score distribution
   - Processing time analytics

## API Integration

The frontend communicates with the backend using a dedicated API client built with Axios:

```typescript
// src/api/client.ts
import axios from "axios";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const apiKey = import.meta.env.VITE_API_KEY || "";

export const apiClient = axios.create({
  baseURL,
  headers: {
    "Content-Type": "application/json",
    "x-api-key": apiKey
  }
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle API errors
    const message = error.response?.data?.error?.message || "Unknown error occurred";
    console.error("API Error:", message);
    return Promise.reject(error);
  }
);
```

## State Management

For most state management needs, React Query provides a robust solution for server state:

```typescript
// src/hooks/useClassification.ts
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { classifyFiles } from "../api/documents";

export function useClassification() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (files: File[]) => {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append("files", file);
      });
      return classifyFiles(formData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["recentClassifications"] });
    }
  });
}
```

For application state, use React Context API for global state and local state hooks for component-specific state.

## Styling Approach

The application uses [Mantine](https://mantine.dev/) for UI components, which provides a comprehensive set of accessible, customizable components with built-in TypeScript support.

```typescript
// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { MantineProvider } from "@mantine/core";
import { Notifications } from "@mantine/notifications";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MantineProvider
      withGlobalStyles
      withNormalizeCSS
    >
      <Notifications />
      <App />
    </MantineProvider>
  </React.StrictMode>
);
```

## Deployment

### Build for Production

```bash
# Using npm
npm run build

# Using yarn
yarn build
```

This creates a `dist` directory with production-ready assets.

### Deployment Options

1. **Vercel Deployment**:

   - Connect your GitHub repository to Vercel
   - Configure environment variables for API URL and key
   - Deploy with default settings

2. **Netlify Deployment**:

   - Connect your GitHub repository to Netlify
   - Set build command to `npm run build` or `yarn build`
   - Set publish directory to `dist`
   - Configure environment variables

3. **Docker Deployment** (alongside backend):
   - Create a Dockerfile in the frontend project
   - Use NGINX to serve static assets
   - Configure NGINX to proxy API requests to the backend

## Example Component Implementation

Here's an example of the `FileUpload` component:

```tsx
// src/components/FileUpload/FileUpload.tsx
import React, { useState } from "react";
import { Group, Text, useMantineTheme, rem } from "@mantine/core";
import { Dropzone, FileWithPath } from "@mantine/dropzone";
import { IconUpload, IconX, IconFile } from "@tabler/icons-react";
import { useClassification } from "../../hooks/useClassification";

interface FileUploadProps {
  onUploadComplete: (results: any[]) => void;
}

export function FileUpload({ onUploadComplete }: FileUploadProps) {
  const theme = useMantineTheme();
  const [files, setFiles] = useState<FileWithPath[]>([]);
  const { mutate, isLoading, error } = useClassification();

  const handleDrop = (acceptedFiles: FileWithPath[]) => {
    setFiles(acceptedFiles);

    // Auto-upload when files are dropped
    if (acceptedFiles.length > 0) {
      mutate(acceptedFiles, {
        onSuccess: (data) => {
          onUploadComplete(data);
          setFiles([]);
        }
      });
    }
  };

  return (
    <Dropzone
      onDrop={handleDrop}
      maxSize={10 * 1024 * 1024} // 10MB max size
      accept={{
        "application/pdf": [".pdf"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
        "image/jpeg": [".jpg", ".jpeg"],
        "image/png": [".png"],
        "text/csv": [".csv"]
      }}
      loading={isLoading}
    >
      <Group
        position="center"
        spacing="xl"
        style={{ minHeight: rem(220), pointerEvents: "none" }}
      >
        <Dropzone.Accept>
          <IconUpload
            size={50}
            stroke={1.5}
            color={theme.colors[theme.primaryColor][theme.colorScheme === "dark" ? 4 : 6]}
          />
        </Dropzone.Accept>
        <Dropzone.Reject>
          <IconX
            size={50}
            stroke={1.5}
            color={theme.colors.red[theme.colorScheme === "dark" ? 4 : 6]}
          />
        </Dropzone.Reject>
        <Dropzone.Idle>
          <IconFile
            size={50}
            stroke={1.5}
          />
        </Dropzone.Idle>

        <div>
          <Text
            size="xl"
            inline
          >
            Drag documents here or click to select files
          </Text>
          <Text
            size="sm"
            color="dimmed"
            inline
            mt={7}
          >
            Attach up to 50 files, each file should not exceed 10MB
          </Text>
          {error && (
            <Text
              color="red"
              size="sm"
              mt={5}
            >
              {error instanceof Error ? error.message : "Upload failed"}
            </Text>
          )}
        </div>
      </Group>
    </Dropzone>
  );
}
```

For the results display component:

```tsx
// src/components/ClassificationResult/ResultCard.tsx
import React from "react";
import { Card, Group, Text, Badge, Progress, Stack } from "@mantine/core";

interface ResultCardProps {
  result: {
    filename: string;
    label: string;
    confidence: number;
    stage_confidences: Record<string, number | null>;
    processing_ms: number;
    warnings: Array<{ code: string; message: string }>;
  };
}

export function ResultCard({ result }: ResultCardProps) {
  const confidenceColor = result.confidence >= 0.9 ? "green" : result.confidence >= 0.7 ? "blue" : result.confidence >= 0.5 ? "yellow" : "red";

  return (
    <Card
      shadow="sm"
      padding="lg"
      radius="md"
      withBorder
    >
      <Card.Section
        p="md"
        bg="gray.1"
      >
        <Group position="apart">
          <Text weight={500}>{result.filename}</Text>
          <Badge color={result.label === "unsure" ? "gray" : "blue"}>{result.label}</Badge>
        </Group>
      </Card.Section>

      <Stack
        spacing="xs"
        mt="md"
      >
        <Group position="apart">
          <Text size="sm">Confidence</Text>
          <Text
            size="sm"
            weight={500}
          >
            {(result.confidence * 100).toFixed(1)}%
          </Text>
        </Group>
        <Progress
          value={result.confidence * 100}
          color={confidenceColor}
          size="lg"
          radius="xl"
        />

        <Text
          size="sm"
          color="dimmed"
          mt="xs"
        >
          Stage Confidences:
        </Text>
        {Object.entries(result.stage_confidences).map(
          ([stage, confidence]) =>
            confidence !== null && (
              <Group
                key={stage}
                position="apart"
                spacing="xs"
              >
                <Text
                  size="xs"
                  color="dimmed"
                >
                  {stage.replace("stage_", "")}
                </Text>
                <Progress
                  value={confidence * 100}
                  color={confidenceColor}
                  size="xs"
                  radius="xl"
                  style={{ width: "70%" }}
                />
                <Text size="xs">{(confidence * 100).toFixed(0)}%</Text>
              </Group>
            )
        )}

        <Text
          size="xs"
          color="dimmed"
          mt="md"
        >
          Processed in {result.processing_ms.toFixed(0)}ms
        </Text>

        {result.warnings.length > 0 && (
          <Stack
            spacing={0}
            mt="xs"
          >
            <Text
              size="xs"
              color="orange"
            >
              Warnings:
            </Text>
            {result.warnings.map((warning, index) => (
              <Text
                key={index}
                size="xs"
                color="dimmed"
              >
                {warning.message}
              </Text>
            ))}
          </Stack>
        )}
      </Stack>
    </Card>
  );
}
```

These examples provide a starting point for implementing the frontend components. The complete implementation would include additional components for job management, navigation, and dashboard visualizations.
