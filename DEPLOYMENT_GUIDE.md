# DTCC Trade Commentary - Render Deployment Guide

## 🚀 Quick Deployment Summary

Your DTCC Trade Commentary application is now ready for deployment to Render! Here's what we've set up:

### ✅ Completed Setup
- ✅ Git repository initialized and pushed to GitHub
- ✅ HTML templates created for Flask web interface
- ✅ Required CSV files (fx.csv, MPC_Dates.csv, IMM_Dates.csv) created
- ✅ Render configuration (render.yaml) updated
- ✅ All dependencies verified and tested
- ✅ Local testing completed successfully

### 📁 Project Structure
```
Trade commentary/
├── dtcc_fetcher.py          # Runs every 1 minute, creates dtcc_trades.csv
├── generate_fx_commentary.py # Generates daily_commentary.md
├── commentary_webapp.py     # Flask web app
├── start_with_fetcher.py    # Combined startup script for Render
├── templates/               # HTML templates
│   ├── index.html          # Dashboard
│   ├── commentary.html     # Commentary display
│   └── error.html          # Error page
├── fx.csv                  # FX conversion rates
├── MPC_Dates.csv           # Central bank meeting dates
├── IMM_Dates.csv           # IMM futures dates
├── dtcc_trades.csv         # Trade data (612 trades loaded)
├── requirements_webapp.txt # Python dependencies
└── render.yaml            # Render deployment config
```

## 🌐 Render Deployment Steps

### Step 1: Sign Up for Render
1. Go to [render.com](https://render.com)
2. Sign up using your GitHub account
3. Connect your GitHub repository

### Step 2: Create Web Service
1. In Render dashboard, click **"New +"** → **"Web Service"**
2. Connect your repository: `Amitr16/trade_commentary`
3. Select branch: `master`

### Step 3: Configure Service
Render will auto-detect the configuration from `render.yaml`:

- **Name**: `dtcc-commentary-webapp`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements_webapp.txt`
- **Start Command**: `python start_with_fetcher.py`
- **Plan**: `Free` (750 hours/month)

### Step 4: Set Environment Variables
In the Render dashboard, go to "Environment" tab and add:

| Key | Value | Description |
|-----|-------|-------------|
| `OPENAI_API_KEY` | `your_openai_api_key` | Required for AI commentary generation |
| `FLASK_ENV` | `production` | Flask environment setting |
| `RENDER` | `true` | Enables Render-specific behavior |

### Step 5: Deploy
1. Click **"Create Web Service"**
2. Wait for build and deployment (5-10 minutes)
3. Your app will be available at: `https://dtcc-commentary-webapp.onrender.com`

## 🔧 How It Works

### Data Flow
1. **DTCC Fetcher** (`dtcc_fetcher.py`):
   - Runs every 1 minute in background
   - Fetches data from DTCC APIs (CFTC, SEC, Canada)
   - Processes and saves to `dtcc_trades.csv`
   - Handles duplicate detection and trade modifications

2. **Commentary Generator** (`generate_fx_commentary.py`):
   - Triggered by web interface "Refresh" button
   - Analyzes latest trade data from CSV
   - Uses OpenAI GPT-4 to generate professional commentary
   - Creates `daily_commentary.md` file

3. **Web Interface** (`commentary_webapp.py`):
   - Flask web app with dashboard and commentary views
   - Auto-refreshes commentary every 5 minutes
   - Displays markdown commentary as HTML
   - Shows top 5 trades by DV01

### Features
- **Real-time Data**: Fetcher runs continuously every minute
- **AI Commentary**: Professional analysis using OpenAI GPT-4
- **Auto-refresh**: Commentary updates automatically
- **Responsive UI**: Clean, modern web interface
- **Error Handling**: Graceful error pages and logging

## 📊 Current Data Status
- **612 trades** loaded in `dtcc_trades.csv`
- **17 currencies** configured in `fx.csv`
- **58 MPC dates** loaded for central bank meetings
- **4 IMM dates** configured for futures contracts

## 🛠️ Local Development

To run locally for testing:

```bash
# Install dependencies
pip install -r requirements_webapp.txt

# Run DTCC fetcher (in one terminal)
python dtcc_fetcher.py

# Run web app (in another terminal)
python commentary_webapp.py

# Access at: http://localhost:5000
```

## 🔍 Monitoring & Logs

### Render Dashboard
- View build logs and runtime logs
- Monitor service health and performance
- Check environment variables

### Application Logs
- DTCC fetcher logs: `dtcc_fetcher.log`
- Flask app logs: Available in Render dashboard
- Commentary generation: Console output

## 💰 Cost Considerations

### Render Free Tier
- **750 hours/month** (about 10.4 hours/day)
- **15-minute sleep** after inactivity
- **512 MB RAM** limit
- **0.1 CPU** limit

### OpenAI API Costs
- **GPT-4**: ~$0.03 per commentary generation
- **Estimated cost**: $1-5/month depending on usage

### Upgrade Options
- **Render Starter**: $7/month for always-on service
- **Render Standard**: $25/month for better performance

## 🚨 Important Notes

### Data Persistence
- Render **does not persist files** between deployments
- CSV files will reset on each deployment
- Consider using external database for production

### Rate Limits
- DTCC API: No official limits, but be respectful
- OpenAI API: Check your usage limits
- Render: Free tier has resource limits

### Security
- **Never commit API keys** to GitHub
- Use Render environment variables for secrets
- Monitor OpenAI API usage and costs

## 🔧 Troubleshooting

### Common Issues

1. **Build Fails**
   - Check `requirements_webapp.txt` for correct dependencies
   - Verify Python version compatibility

2. **Startup Fails**
   - Check `start_with_fetcher.py` and port configuration
   - Verify environment variables are set

3. **No Commentary Generated**
   - Ensure `OPENAI_API_KEY` is set correctly
   - Check if `dtcc_trades.csv` has data
   - Verify OpenAI API credits

4. **DTCC Fetcher Not Working**
   - Check network connectivity
   - Verify DTCC API endpoints are accessible
   - Review fetcher logs

### Getting Help
- Check Render logs in dashboard
- Review application logs in `dtcc_fetcher.log`
- Test locally first before deploying
- Use Render's support documentation

## 🎯 Next Steps

1. **Deploy to Render** using the steps above
2. **Set OpenAI API key** in environment variables
3. **Test the application** by generating commentary
4. **Monitor performance** and adjust as needed
5. **Consider database integration** for production use

## 📞 Support

- **Render Documentation**: [render.com/docs](https://render.com/docs)
- **OpenAI API Documentation**: [platform.openai.com](https://platform.openai.com)
- **GitHub Repository**: [github.com/Amitr16/trade_commentary](https://github.com/Amitr16/trade_commentary)

---

**Ready to deploy!** 🚀 Your DTCC Trade Commentary application is fully configured and tested. Follow the deployment steps above to get it live on Render.
