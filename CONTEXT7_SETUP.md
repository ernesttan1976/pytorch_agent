# Context7 MCP Setup Guide

This guide explains how to set up Context7 MCP (Model Context Protocol) server to get the latest library documentation and API references.

## What is Context7 MCP?

Context7 MCP provides real-time, version-specific documentation and code examples directly to your AI coding assistant. It ensures you always have access to the latest library documentation, API changes, and best practices.

## Setup Instructions

### 1. Prerequisites

- Node.js version 18 or higher
- A Context7 account (sign up at https://context7.com/dashboard)

### 2. Install Context7 MCP

```bash
npm install -g @upstash/context7-mcp@latest
```

### 3. Get Your API Key

1. Sign up or log in at https://context7.com/dashboard
2. Navigate to your API keys section
3. Create a new API key
4. Copy the API key (you'll need it for configuration)

### 4. Configure in Cursor

#### Option A: Via Cursor Settings UI

1. Open Cursor Settings (`File` -> `Preferences` -> `Settings` or `Ctrl+,`)
2. Navigate to `Cursor Settings` -> `MCP` -> `Add new global MCP server`
3. Add the following configuration:

```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp",
      "headers": {
        "CONTEXT7_API_KEY": "YOUR_API_KEY_HERE"
      }
    }
  }
}
```

Replace `YOUR_API_KEY_HERE` with your actual Context7 API key.

#### Option B: Manual Configuration File

1. Locate your Cursor MCP configuration file:
   - Windows: `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
   - macOS: `~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
   - Linux: `~/.config/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

2. Add the Context7 configuration to the file:

```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp",
      "headers": {
        "CONTEXT7_API_KEY": "YOUR_API_KEY_HERE"
      }
    }
  }
}
```

3. Restart Cursor for changes to take effect.

### 5. Configure Automatic Usage (Optional but Recommended)

To automatically use Context7 for code-related queries:

1. Go to `Cursor Settings` -> `Rules`
2. Add a rule:
   ```
   Always use Context7 MCP when I need code generation, setup or configuration steps, 
   or library/API documentation. This means you should automatically use the Context7 
   MCP tools to resolve library id and get library docs without me having to explicitly ask.
   ```

This ensures Context7 is used automatically for relevant queries, providing up-to-date documentation seamlessly.

### 6. Verify Installation

After setup, Context7 MCP will be available automatically. When you ask questions about libraries, APIs, or need documentation, the assistant will use Context7 to get the latest information.

## Updating Context7 MCP

To ensure you're using the latest version:

```bash
npm install -g @upstash/context7-mcp@latest
```

## Troubleshooting

- **MCP not working**: Ensure Node.js 18+ is installed and the API key is correct
- **Outdated documentation**: Update to the latest version of Context7 MCP
- **Configuration issues**: Verify the JSON syntax in your MCP configuration file

For more help, see: https://context7.com/docs/resources/troubleshooting

## Benefits

- **Always up-to-date**: Get the latest library documentation and API changes
- **Version-specific**: Documentation matches your installed library versions
- **Real-time**: No need to wait for model training on new documentation
- **Automatic**: Works seamlessly in the background when configured properly

