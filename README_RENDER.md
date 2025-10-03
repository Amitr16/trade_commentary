# DTCC Commentary Web App - Render Deployment Guide

This guide will help you deploy the DTCC Commentary Web App to Render.

## Prerequisites

1. **GitHub Repository**: Push your code to a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **OpenAI API Key**: Get your API key from [OpenAI](https://platform.openai.com)

## Files for Deployment

The following files are required for Render deployment:

- `render.yaml` - Render configuration
- `start_render.py` - Startup script for Render
- `commentary_webapp.py` - Main Flask application
- `requirements_webapp.txt` - Python dependencies
- `templates/` - HTML templates directory
- `dtcc_trades.csv` - Trade data (will be empty initially)
- `fx.csv` - FX rates file
- `MPC_Dates.csv` - MPC dates configuration
- `IMM_Dates.csv` - IMM dates configuration

## Deployment Steps

### 1. Push to GitHub

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Create Render Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select your repository and branch (usually `main`)

### 3. Configure Service Settings

- **Name**: `dtcc-commentary-webapp`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements_webapp.txt`
- **Start Command**: `python start_render.py`

### 4. Set Environment Variables

In the Render dashboard, go to "Environment" and add:

- `OPENAI_API_KEY` = `your_openai_api_key_here`
- `FLASK_ENV` = `production`
- `PORT` = `10000` (Render will override this)

### 5. Deploy

Click "Create Web Service" and wait for deployment to complete.

## Post-Deployment Setup

After deployment, you'll need to:

1. **Add Trade Data**: The app starts with empty `dtcc_trades.csv`
2. **Generate Initial Commentary**: Click refresh to run the AI commentary generation
3. **Set up Data Updates**: Configure your `dtcc_fetcher.py` to update the CSV file

## Data Updates

Since Render doesn't persist files between deployments, you have a few options:

### Option 1: External Data Source
- Store `dtcc_trades.csv` in a cloud storage (S3, Google Drive, etc.)
- Modify the app to fetch data from external source

### Option 2: Database Integration
- Use a database like PostgreSQL (Render provides this)
- Modify the app to read from database instead of CSV

### Option 3: Scheduled Updates
- Use Render's Cron Jobs to run `dtcc_fetcher.py` periodically
- Update the CSV file regularly

## Environment Variables

Required environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key for commentary generation
- `FLASK_ENV` - Set to `production` for Render
- `PORT` - Automatically set by Render

## Troubleshooting

### Common Issues

1. **Build Fails**: Check `requirements_webapp.txt` for correct dependencies
2. **Startup Fails**: Check `start_render.py` and port configuration
3. **No Data**: Ensure `dtcc_trades.csv` exists and has data
4. **OpenAI Errors**: Verify `OPENAI_API_KEY` is set correctly

### Logs

Check Render logs in the dashboard for detailed error messages.

## Local Testing

Test the production setup locally:

```bash
# Set environment variables
export RENDER=true
export PORT=10000
export OPENAI_API_KEY=your_key_here

# Run the app
python start_render.py
```

## Cost Considerations

- **Free Tier**: 750 hours/month, sleeps after 15 minutes of inactivity
- **Paid Plans**: Start at $7/month for always-on service
- **OpenAI API**: Pay per use for commentary generation

## Security Notes

- Never commit API keys to GitHub
- Use Render's environment variables for sensitive data
- Consider rate limiting for public access
- Monitor OpenAI API usage and costs
