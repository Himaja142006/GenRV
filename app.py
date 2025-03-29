from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

def generate_distribution(dist_type, params):
    size = 1000  # Sample size for plotting
    
    if dist_type == 'normal':
        mu = float(params['mu'])
        sigma = float(params['sigma'])
        sample = np.random.normal(mu, sigma)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, size)
        pdf = stats.norm.pdf(x, mu, sigma)
        cdf = stats.norm.cdf(x, mu, sigma)
        explanation = f"Generated from Normal distribution with mean μ={mu} and standard deviation σ={sigma}"
        
    elif dist_type == 'uniform':
        a = float(params['a'])
        b = float(params['b'])
        sample = np.random.uniform(a, b)
        x = np.linspace(a - 0.1*(b-a), b + 0.1*(b-a), size)
        pdf = stats.uniform.pdf(x, a, b-a)
        cdf = stats.uniform.cdf(x, a, b-a)
        explanation = f"Generated from Uniform distribution with minimum a={a} and maximum b={b}"
        
    elif dist_type == 'poisson':
        lam = float(params['lambda'])
        sample = np.random.poisson(lam)
        x = np.arange(0, max(20, int(lam*3)))
        pdf = stats.poisson.pmf(x, lam)
        cdf = stats.poisson.cdf(x, lam)
        explanation = f"Generated from Poisson distribution with rate λ={lam}"
        
    elif dist_type == 'rayleigh':
        scale = float(params['scale'])
        sample = np.random.rayleigh(scale)
        x = np.linspace(0, scale*5, size)
        pdf = stats.rayleigh.pdf(x, scale=scale)
        cdf = stats.rayleigh.cdf(x, scale=scale)
        explanation = f"Generated from Rayleigh distribution with scale σ={scale}"
        
    elif dist_type == 'laplace':
        loc = float(params['loc'])
        scale = float(params['scale'])
        sample = np.random.laplace(loc, scale)
        x = np.linspace(loc - 6*scale, loc + 6*scale, size)
        pdf = stats.laplace.pdf(x, loc=loc, scale=scale)
        cdf = stats.laplace.cdf(x, loc=loc, scale=scale)
        explanation = f"Generated from Laplace distribution with location μ={loc} and scale b={scale}"
        
    elif dist_type == 'exponential':
        scale = float(params['scale'])
        sample = np.random.exponential(scale)
        x = np.linspace(0, scale*5, size)
        pdf = stats.expon.pdf(x, scale=scale)
        cdf = stats.expon.cdf(x, scale=scale)
        explanation = f"Generated from Exponential distribution with scale 1/λ={scale}"
        
    elif dist_type == 'binomial':
        n = int(params['n'])
        p = float(params['p'])
        sample = np.random.binomial(n, p)
        x = np.arange(0, n+1)
        pdf = stats.binom.pmf(x, n, p)
        cdf = stats.binom.cdf(x, n, p)
        explanation = f"Generated from Binomial distribution with n={n} trials and success probability p={p}"
    
    return sample, x, pdf, cdf, explanation

def calculate_statistics(dist_type, params):
    if dist_type == 'normal':
        mu = float(params['mu'])
        sigma = float(params['sigma'])
        moments = [mu, mu**2 + sigma**2, mu**3 + 3*mu*sigma**2, mu**4 + 6*mu**2*sigma**2 + 3*sigma**4]
        central_moments = [0, sigma**2, 0, 3*sigma**4]
        skewness = 0
        kurtosis = 3
        mgf = f"exp(μt + σ²t²/2)"
        cf = f"exp(iμt - σ²t²/2)"
        
    elif dist_type == 'uniform':
        a = float(params['a'])
        b = float(params['b'])
        moments = [(a+b)/2, (a**2 + a*b + b**2)/3, (a**3 + a**2*b + a*b**2 + b**3)/4,
                  (a**4 + a**3*b + a**2*b**2 + a*b**3 + b**4)/5]
        central_moments = [0, (b-a)**2/12, 0, (b-a)**4/80]
        skewness = 0
        kurtosis = 1.8
        mgf = f"(exp(bt) - exp(at))/(t(b-a))"
        cf = f"(exp(ibt) - exp(iat))/(it(b-a))"
        
    elif dist_type == 'poisson':
        lam = float(params['lambda'])
        moments = [lam, lam + lam**2, lam + 3*lam**2 + lam**3, lam + 7*lam**2 + 6*lam**3 + lam**4]
        central_moments = [0, lam, lam, lam + 3*lam**2]
        skewness = 1/np.sqrt(lam)
        kurtosis = 3 + 1/lam
        mgf = f"exp(λ(exp(t) - 1))"
        cf = f"exp(λ(exp(it) - 1))"
        
    elif dist_type == 'rayleigh':
        scale = float(params['scale'])
        moments = [scale*np.sqrt(np.pi/2), 2*scale**2, 3*scale**3*np.sqrt(np.pi/2),
                  8*scale**4]
        central_moments = [0, (4-np.pi)*scale**2/2, (2*np.sqrt(np.pi)*(np.pi-3))*scale**3/4,
                         ((-6*np.pi**2 + 24*np.pi - 16)*scale**4)/4]
        skewness = 2*np.sqrt(np.pi)*(np.pi-3)/(4-np.pi)**(3/2)
        kurtosis = 3 + ((-6*np.pi**2 + 24*np.pi - 16)/(4-np.pi)**2)
        mgf = "Not available in closed form"
        cf = f"exp(-σ²t²/2)(1 + iσt√(π/2))"
        
    elif dist_type == 'laplace':
        loc = float(params['loc'])
        scale = float(params['scale'])
        moments = [loc, loc**2 + 2*scale**2, loc**3 + 6*loc*scale**2,
                  loc**4 + 12*loc**2*scale**2 + 24*scale**4]
        central_moments = [0, 2*scale**2, 0, 24*scale**4]
        skewness = 0
        kurtosis = 6
        mgf = f"exp(μt)/(1 - b²t²)"
        cf = f"exp(iμt)/(1 + b²t²)"
        
    elif dist_type == 'exponential':
        scale = float(params['scale'])
        moments = [scale, 2*scale**2, 6*scale**3, 24*scale**4]
        central_moments = [0, scale**2, 2*scale**3, 9*scale**4]
        skewness = 2
        kurtosis = 9
        mgf = f"1/(1 - λt)"
        cf = f"1/(1 - iλt)"
        
    elif dist_type == 'binomial':
        n = int(params['n'])
        p = float(params['p'])
        q = 1 - p
        moments = [n*p, n*p*(1+p), n*p*(1+3*p+p**2), n*p*(1+7*p+6*p**2+p**3)]
        central_moments = [0, n*p*q, n*p*q*(q-p), n*p*q*(1-6*p*q)]
        skewness = (q-p)/np.sqrt(n*p*q)
        kurtosis = 3 + (1-6*p*q)/(n*p*q)
        mgf = f"(q + p*exp(t))^n"
        cf = f"(q + p*exp(it))^n"
    
    return {
        'moments': moments,
        'central_moments': central_moments,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mgf': mgf,
        'cf': cf
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        dist_type = data['distribution']
        params = data['params']
        
        sample, x, pdf, cdf, explanation = generate_distribution(dist_type, params)
        statistics = calculate_statistics(dist_type, params)
        
        return jsonify({
            'sample': float(sample),
            'explanation': explanation,
            'statistics': statistics,
            'pdf_data': {
                'x': x.tolist(),
                'y': pdf.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'PDF/PMF'
            },
            'cdf_data': {
                'x': x.tolist(),
                'y': cdf.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'CDF'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
