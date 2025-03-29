# GenRV - Random Variable Generator

A powerful web application for generating and analyzing random variables with multilingual speech support.

## Features

- Generate random variables from various distributions:
  - Normal (Gaussian)
  - Uniform
  - Poisson
  - Rayleigh
  - Laplace
  - Exponential
  - Binomial

- Multilingual voice commands support:
  - English
  - Telugu (తెలుగు)
  - Tamil (தமிழ்)
  - Hindi (हिंदी)
  - Kannada (ಕನ್ನಡ)

- Statistical Analysis:
  - Moments about origin
  - Central moments
  - Moment generating function
  - Characteristic function
  - Skewness and Kurtosis
  - PDF and CDF plots

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genrv.git
cd genrv
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
flask run
```

## Deployment

The application is ready for deployment on platforms like Heroku:

1. Create a Heroku account and install Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Deploy:
```bash
git push heroku main
```

## Desktop Application

To convert this web application into a desktop application:

1. Install additional requirements:
```bash
pip install pyinstaller
```

2. Build the desktop application:
```bash
pyinstaller --onefile --add-data "templates:templates" app.py
```

The executable will be available in the `dist` directory.

## License

MIT License

## Author

Your Name
