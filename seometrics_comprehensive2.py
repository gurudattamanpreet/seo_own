"""
SEOMetrics Professional - Production-Ready SEO Analysis Tool
=============================================================
A comprehensive SEO analysis tool with industry-standard metrics,
advanced crawling capabilities, and detailed issue categorization.

Author: SEOMetrics Team
Version: 2.1.0 - FastMCP Integration & AI Recommendations
License: MIT
"""

import asyncio
import json
import os
import re
import socket
import ssl
import time
import urllib.parse
import hashlib
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET
import gzip
import io

# Core Web Framework
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

# HTTP & Web Scraping
import requests
from bs4 import BeautifulSoup, Comment
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Content Analysis
import textstat
from collections import OrderedDict

# Advanced Features - Optional but Recommended
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not available - install: pip install playwright && playwright install")

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    print("⚠️  Whois not available - install: pip install python-whois")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available - install: pip install pillow")

try:
    import nltk
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('punkt')
    except:
        nltk.download('punkt', quiet=True)
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not available - install: pip install textblob nltk")

# Groq AI Integration - Using environment variables
GROQ_AVAILABLE = False
groq_client = None

try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            test_response = groq_client.models.list()
            GROQ_AVAILABLE = True
            print("✓ Groq AI initialized successfully")
        except Exception as init_error:
            GROQ_AVAILABLE = False
            groq_client = None
            print(f"⚠️  Groq AI initialization failed: {str(init_error)}")
    else:
        print("⚠️  GROQ_API_KEY not found in environment variables")
        
except ImportError:
    print("⚠️  Groq not installed - install: pip install groq")
    GROQ_AVAILABLE = False
    groq_client = None
except Exception as e:
    print(f"⚠️  Groq initialization failed: {str(e)}")
    GROQ_AVAILABLE = False
    groq_client = None

# Google PageSpeed Insights API - Using environment variable
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY", "")
PAGESPEED_API_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

# Initialize FastAPI
app = FastAPI(
    title="SEOMetrics Professional",
    description="Production-Ready SEO Analysis Tool",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates directory
templates = Jinja2Templates(directory="templates")
if not os.path.exists("templates"):
    os.makedirs("templates")

# =======================
# ISSUE DEFINITIONS
# =======================

class SEOIssueDefinitions:
    """Complete SEO issue definitions based on the provided PDF"""
    
    # Categorized issues from the PDF
    ISSUES = {
        # ERRORS (Critical issues that must be fixed)
        1: {"title": "5xx Server Errors", "severity": "error", "category": "server"},
        2: {"title": "4xx Client Errors", "severity": "error", "category": "server"},
        3: {"title": "Title Tag Missing or Empty", "severity": "error", "category": "meta"},
        4: {"title": "Blocked from Crawling", "severity": "error", "category": "crawlability"},
        6: {"title": "Duplicate Title Tags", "severity": "error", "category": "meta"},
        7: {"title": "Duplicate Content", "severity": "error", "category": "content"},
        8: {"title": "Broken Internal Links", "severity": "error", "category": "links"},
        9: {"title": "Pages Not Crawled", "severity": "error", "category": "crawlability"},
        10: {"title": "DNS Resolution Issue", "severity": "error", "category": "server"},
        11: {"title": "Cannot Open Page URL", "severity": "error", "category": "server"},
        13: {"title": "Broken Internal Images", "severity": "error", "category": "media"},
        15: {"title": "Duplicate Meta Descriptions", "severity": "error", "category": "meta"},
        16: {"title": "Invalid robots.txt Format", "severity": "error", "category": "crawlability"},
        17: {"title": "Invalid sitemap.xml Format", "severity": "error", "category": "crawlability"},
        18: {"title": "Incorrect Pages in sitemap.xml", "severity": "error", "category": "crawlability"},
        19: {"title": "WWW Resolve Issues", "severity": "error", "category": "server"},
        20: {"title": "Viewport Not Configured", "severity": "error", "category": "mobile"},
        21: {"title": "Large HTML Page Size", "severity": "error", "category": "performance"},
        22: {"title": "Missing Canonical Tags in AMP", "severity": "error", "category": "amp"},
        26: {"title": "Non-Secure Pages", "severity": "error", "category": "security"},
        27: {"title": "Certificate Expiration", "severity": "error", "category": "security"},
        28: {"title": "Old Security Protocol Version", "severity": "error", "category": "security"},
        29: {"title": "Certificate Registered to Incorrect Name", "severity": "error", "category": "security"},
        30: {"title": "Mixed Content Issues", "severity": "error", "category": "security"},
        32: {"title": "No Canonical or 301 from HTTP", "severity": "error", "category": "redirects"},
        33: {"title": "Redirect Chains and Loops", "severity": "error", "category": "redirects"},
        34: {"title": "AMP Pages with HTML Issues", "severity": "error", "category": "amp"},
        35: {"title": "AMP Pages with Style Issues", "severity": "error", "category": "amp"},
        36: {"title": "AMP Pages with Templating Issues", "severity": "error", "category": "amp"},
        38: {"title": "Broken Canonical URLs", "severity": "error", "category": "meta"},
        39: {"title": "Multiple Canonical URLs", "severity": "error", "category": "meta"},
        40: {"title": "Meta Refresh Redirects", "severity": "error", "category": "redirects"},
        41: {"title": "Broken Internal JS/CSS Files", "severity": "error", "category": "resources"},
        42: {"title": "Insecure Encryption Algorithms", "severity": "error", "category": "security"},
        43: {"title": "Sitemap File Too Large", "severity": "error", "category": "crawlability"},
        44: {"title": "Malformed Links", "severity": "error", "category": "links"},
        45: {"title": "Structured Data Markup Errors", "severity": "error", "category": "structured_data"},
        46: {"title": "Viewport Width Not Set", "severity": "error", "category": "mobile"},
        103: {"title": "Missing H1 Tag", "severity": "error", "category": "headings"},
        126: {"title": "HTTPS Encryption Not Used", "severity": "error", "category": "security"},
        
        # WARNINGS (Important issues that should be fixed)
        12: {"title": "Broken External Links", "severity": "warning", "category": "links"},
        14: {"title": "Broken External Images", "severity": "warning", "category": "media"},
        31: {"title": "Links to HTTP Pages from HTTPS", "severity": "warning", "category": "security"},
        101: {"title": "Title Element Too Short", "severity": "warning", "category": "meta"},
        102: {"title": "Title Element Too Long", "severity": "warning", "category": "meta"},
        104: {"title": "Multiple H1 Tags", "severity": "warning", "category": "headings"},
        105: {"title": "Duplicate H1 and Title", "severity": "warning", "category": "headings"},
        106: {"title": "Missing Meta Description", "severity": "warning", "category": "meta"},
        108: {"title": "Too Many On-Page Links", "severity": "warning", "category": "links"},
        109: {"title": "Temporary Redirects", "severity": "warning", "category": "redirects"},
        110: {"title": "Missing ALT Attributes", "severity": "warning", "category": "media"},
        111: {"title": "Slow Page Load Speed", "severity": "warning", "category": "performance"},
        112: {"title": "Low Text to HTML Ratio", "severity": "warning", "category": "content"},
        113: {"title": "Too Many URL Parameters", "severity": "warning", "category": "urls"},
        114: {"title": "Missing Hreflang/Lang Attributes", "severity": "warning", "category": "international"},
        115: {"title": "Encoding Not Declared", "severity": "warning", "category": "technical"},
        116: {"title": "Doctype Not Declared", "severity": "warning", "category": "technical"},
        117: {"title": "Low Word Count", "severity": "warning", "category": "content"},
        120: {"title": "Incompatible Plugins Used", "severity": "warning", "category": "technical"},
        121: {"title": "Frames Used", "severity": "warning", "category": "technical"},
        122: {"title": "Underscores in URL", "severity": "warning", "category": "urls"},
        123: {"title": "Nofollow in Internal Links", "severity": "warning", "category": "links"},
        124: {"title": "Sitemap Not in robots.txt", "severity": "warning", "category": "crawlability"},
        125: {"title": "Sitemap.xml Not Found", "severity": "warning", "category": "crawlability"},
        127: {"title": "No SNI Support", "severity": "warning", "category": "security"},
        128: {"title": "HTTP URLs in Sitemap for HTTPS", "severity": "warning", "category": "crawlability"},
        129: {"title": "Uncompressed Pages", "severity": "warning", "category": "performance"},
        130: {"title": "Disallowed Internal Resources", "severity": "warning", "category": "crawlability"},
        131: {"title": "Uncompressed JS/CSS Files", "severity": "warning", "category": "performance"},
        132: {"title": "Uncached JS/CSS Files", "severity": "warning", "category": "performance"},
        133: {"title": "Large JS/CSS Total Size", "severity": "warning", "category": "performance"},
        134: {"title": "Too Many JS/CSS Files", "severity": "warning", "category": "performance"},
        135: {"title": "Unminified JS/CSS Files", "severity": "warning", "category": "performance"},
        136: {"title": "Too Long URLs", "severity": "warning", "category": "urls"},
        201: {"title": "URLs Too Long", "severity": "warning", "category": "urls"},
        208: {"title": "High Document Interactive Time", "severity": "warning", "category": "performance"},
        209: {"title": "Blocked by X-Robots-Tag", "severity": "warning", "category": "crawlability"},
        210: {"title": "Disallowed External Resources", "severity": "warning", "category": "crawlability"},
        211: {"title": "Broken External JS/CSS", "severity": "warning", "category": "resources"},
        212: {"title": "High Page Crawl Depth", "severity": "warning", "category": "structure"},
        213: {"title": "Pages with One Internal Link", "severity": "warning", "category": "links"},
        
        # NOTICES (Minor issues or suggestions)
        137: {"title": "Llms.txt Not Found", "severity": "notice", "category": "crawlability"},
        202: {"title": "Nofollow in External Links", "severity": "notice", "category": "links"},
        203: {"title": "Robots.txt Not Found", "severity": "notice", "category": "crawlability"},
        205: {"title": "No HSTS Support", "severity": "notice", "category": "security"},
        206: {"title": "Orphaned Pages", "severity": "notice", "category": "structure"},
        207: {"title": "Orphaned Sitemap Pages", "severity": "notice", "category": "crawlability"},
        214: {"title": "Permanent Redirects", "severity": "notice", "category": "redirects"},
        215: {"title": "Resources as Page Links", "severity": "notice", "category": "links"},
        216: {"title": "Links with No Anchor Text", "severity": "notice", "category": "links"},
        217: {"title": "Non-Descriptive Anchor Text", "severity": "notice", "category": "links"},
        218: {"title": "External 403 Status", "severity": "notice", "category": "links"},
        219: {"title": "Llms.txt Formatting Issues", "severity": "notice", "category": "crawlability"},
        220: {"title": "Too Much Content", "severity": "notice", "category": "content"},
        221: {"title": "Outdated Content", "severity": "notice", "category": "content"},
        222: {"title": "Low Semantic HTML Usage", "severity": "notice", "category": "technical"},
        223: {"title": "Content Not Optimized", "severity": "notice", "category": "content"},
    }
    
    @classmethod
    def get_issue_details(cls, code: int) -> Dict:
        """Get detailed information about an issue"""
        return cls.ISSUES.get(code, {
            "title": f"Unknown Issue #{code}",
            "severity": "notice",
            "category": "unknown"
        })
    
    @classmethod
    def get_severity_priority(cls, severity: str) -> int:
        """Get priority based on severity"""
        priorities = {"error": 1, "warning": 2, "notice": 3}
        return priorities.get(severity, 4)


# =======================
# ADVANCED CRAWLERS
# =======================

class RobotsTxtParser:
    """Parse and validate robots.txt"""
    
    @staticmethod
    async def parse(url: str) -> Dict:
        """Parse robots.txt file"""
        try:
            robots_url = urllib.parse.urljoin(url, '/robots.txt')
            response = requests.get(robots_url, timeout=5)
            
            if response.status_code == 404:
                return {"exists": False, "valid": False, "sitemaps": [], "rules": {}}
            
            if response.status_code != 200:
                return {"exists": True, "valid": False, "error": f"Status {response.status_code}"}
            
            content = response.text
            lines = content.split('\n')
            
            sitemaps = []
            rules = defaultdict(list)
            current_agent = '*'
            
            for line in lines:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                if ':' not in line:
                    continue
                    
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'user-agent':
                    current_agent = value
                elif key == 'disallow':
                    rules[current_agent].append(('disallow', value))
                elif key == 'allow':
                    rules[current_agent].append(('allow', value))
                elif key == 'sitemap':
                    sitemaps.append(value)
                elif key == 'crawl-delay':
                    rules[current_agent].append(('crawl-delay', value))
            
            return {
                "exists": True,
                "valid": True,
                "sitemaps": sitemaps,
                "rules": dict(rules),
                "content": content
            }
            
        except Exception as e:
            return {"exists": False, "valid": False, "error": str(e)}


class SitemapParser:
    """Parse and validate XML sitemaps"""
    
    @staticmethod
    async def parse(url: str) -> Dict:
        """Parse sitemap.xml"""
        try:
            sitemap_urls = [
                urllib.parse.urljoin(url, '/sitemap.xml'),
                urllib.parse.urljoin(url, '/sitemap_index.xml'),
                urllib.parse.urljoin(url, '/sitemap-index.xml')
            ]
            
            for sitemap_url in sitemap_urls:
                response = requests.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    return await SitemapParser._parse_sitemap_content(response.content, sitemap_url)
            
            return {"exists": False, "valid": False, "urls": [], "error": "Sitemap not found"}
            
        except Exception as e:
            return {"exists": False, "valid": False, "error": str(e)}
    
    @staticmethod
    async def _parse_sitemap_content(content: bytes, sitemap_url: str) -> Dict:
        """Parse sitemap content"""
        try:
            # Check if gzipped
            if content[:2] == b'\x1f\x8b':
                content = gzip.decompress(content)
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Remove namespace
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}')[1]
            
            urls = []
            sitemap_index = False
            
            # Check if it's a sitemap index
            if root.tag == 'sitemapindex':
                sitemap_index = True
                for sitemap in root.findall('sitemap'):
                    loc = sitemap.find('loc')
                    if loc is not None:
                        urls.append(loc.text)
            else:
                # Regular sitemap
                for url_elem in root.findall('url'):
                    loc = url_elem.find('loc')
                    if loc is not None:
                        url_data = {"loc": loc.text}
                        
                        lastmod = url_elem.find('lastmod')
                        if lastmod is not None:
                            url_data['lastmod'] = lastmod.text
                        
                        priority = url_elem.find('priority')
                        if priority is not None:
                            url_data['priority'] = float(priority.text)
                        
                        changefreq = url_elem.find('changefreq')
                        if changefreq is not None:
                            url_data['changefreq'] = changefreq.text
                        
                        urls.append(url_data)
            
            # Check file size
            file_size = len(content)
            size_mb = file_size / (1024 * 1024)
            
            return {
                "exists": True,
                "valid": True,
                "is_index": sitemap_index,
                "urls": urls[:100],  # Limit for performance
                "total_urls": len(urls),
                "size_mb": round(size_mb, 2),
                "too_large": size_mb > 50,
                "location": sitemap_url
            }
            
        except ET.ParseError as e:
            return {"exists": True, "valid": False, "error": f"XML Parse Error: {str(e)}"}
        except Exception as e:
            return {"exists": True, "valid": False, "error": str(e)}


class SecurityChecker:
    """Check security-related issues"""
    
    @staticmethod
    async def check_ssl(url: str) -> Dict:
        """Check SSL certificate and security"""
        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname
            port = parsed.port or 443
            
            if not url.startswith('https'):
                return {
                    "https": False,
                    "valid": False,
                    "error": "Not using HTTPS"
                }
            
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and get certificate
            with socket.create_connection((hostname, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
            
            # Parse certificate
            not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            days_until_expiry = (not_after - datetime.now()).days
            
            # Check certificate name
            cert_hostname = cert['subject'][0][0][1] if cert.get('subject') else None
            
            # Check protocol version
            old_protocols = ['TLSv1', 'TLSv1.1', 'SSLv2', 'SSLv3']
            is_old_protocol = version in old_protocols
            
            return {
                "https": True,
                "valid": True,
                "expires_in_days": days_until_expiry,
                "expiring_soon": days_until_expiry < 30,
                "expired": days_until_expiry < 0,
                "protocol_version": version,
                "old_protocol": is_old_protocol,
                "cipher": cipher[0] if cipher else None,
                "certificate_hostname": cert_hostname,
                "hostname_match": cert_hostname == hostname
            }
            
        except ssl.SSLError as e:
            return {"https": True, "valid": False, "error": f"SSL Error: {str(e)}"}
        except Exception as e:
            return {"https": False, "valid": False, "error": str(e)}
    
    @staticmethod
    async def check_security_headers(response_headers: dict) -> Dict:
        """Check security headers"""
        security_headers = {
            'Strict-Transport-Security': 'hsts',
            'X-Content-Type-Options': 'x_content_type',
            'X-Frame-Options': 'x_frame',
            'X-XSS-Protection': 'x_xss',
            'Content-Security-Policy': 'csp',
            'Referrer-Policy': 'referrer_policy'
        }
        
        results = {}
        for header, key in security_headers.items():
            results[key] = header.lower() in [h.lower() for h in response_headers.keys()]
        
        results['score'] = sum(1 for v in results.values() if v) / len(results) * 100
        
        return results


class ContentAnalyzer:
    """Analyze content quality and SEO metrics"""
    
    @staticmethod
    def extract_clean_text(soup: BeautifulSoup) -> str:
        """Extract clean text from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def analyze_readability(text: str) -> Dict:
        """Analyze text readability"""
        if not text or len(text) < 100:
            return {
                "flesch_score": 0,
                "flesch_grade": "N/A",
                "interpretation": "Insufficient content",
                "avg_sentence_length": 0,
                "syllables_per_word": 0
            }
        
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_grade = textstat.flesch_kincaid_grade(text)
            
            # Interpretation
            if flesch_score >= 90:
                interpretation = "Very Easy (5th grade)"
            elif flesch_score >= 80:
                interpretation = "Easy (6th grade)"
            elif flesch_score >= 70:
                interpretation = "Fairly Easy (7th grade)"
            elif flesch_score >= 60:
                interpretation = "Standard (8-9th grade)"
            elif flesch_score >= 50:
                interpretation = "Fairly Difficult (10-12th grade)"
            elif flesch_score >= 30:
                interpretation = "Difficult (College)"
            else:
                interpretation = "Very Difficult (Graduate)"
            
            return {
                "flesch_score": round(flesch_score, 1),
                "flesch_grade": round(flesch_grade, 1),
                "interpretation": interpretation,
                "avg_sentence_length": textstat.avg_sentence_length(text),
                "syllables_per_word": textstat.avg_syllables_per_word(text)
            }
            
        except Exception:
            return {
                "flesch_score": 60,
                "flesch_grade": "N/A",
                "interpretation": "Standard",
                "avg_sentence_length": 0,
                "syllables_per_word": 0
            }
    
    @staticmethod
    def analyze_keywords(text: str, soup: BeautifulSoup) -> Dict:
        """Analyze keyword density and usage"""
        if not text:
            return {"keywords": {}, "density": {}}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 
                      'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
                      'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
                      'can', 'shall', 'to', 'of', 'in', 'for', 'with', 'by', 'from',
                      'about', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again'}
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not filtered_words:
            return {"keywords": {}, "density": {}}
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        total_words = len(filtered_words)
        
        # Get top keywords with density
        top_keywords = {}
        for word, count in word_freq.most_common(10):
            density = (count / total_words) * 100
            top_keywords[word] = {
                "count": count,
                "density": round(density, 2)
            }
        
        # Check keyword stuffing
        max_density = max(kw["density"] for kw in top_keywords.values()) if top_keywords else 0
        keyword_stuffing = max_density > 3.0
        
        return {
            "top_keywords": top_keywords,
            "total_words": total_words,
            "unique_words": len(set(filtered_words)),
            "keyword_stuffing_risk": keyword_stuffing,
            "max_keyword_density": round(max_density, 2)
        }
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict:
        """Analyze content sentiment"""
        if not TEXTBLOB_AVAILABLE or not text:
            return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0}
        
        try:
            # Limit text length for performance
            sample_text = text[:5000]
            blob = TextBlob(sample_text)
            
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "interpretation": f"The content appears {sentiment} with {'high' if subjectivity > 0.5 else 'low'} subjectivity"
            }
            
        except Exception:
            return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0}


class PerformanceAnalyzer:
    """Analyze page performance metrics"""
    
    @staticmethod
    async def measure_load_time(url: str) -> Dict:
        """Measure page load time and resources"""
        try:
            start_time = time.time()
            
            # Make request
            response = requests.get(url, timeout=10)
            
            # Basic load time
            load_time = (time.time() - start_time) * 1000
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html5lib')
            
            # Count resources
            scripts = len(soup.find_all('script'))
            stylesheets = len(soup.find_all('link', rel='stylesheet'))
            images = len(soup.find_all('img'))
            iframes = len(soup.find_all('iframe'))
            
            # Calculate sizes
            html_size = len(response.content)
            
            return {
                "load_time_ms": round(load_time, 2),
                "html_size_kb": round(html_size / 1024, 2),
                "scripts": scripts,
                "stylesheets": stylesheets,
                "images": images,
                "iframes": iframes,
                "total_resources": scripts + stylesheets + images + iframes,
                "performance_grade": PerformanceAnalyzer._calculate_grade(load_time)
            }
            
        except Exception as e:
            return {"error": str(e), "load_time_ms": 0}
    
    @staticmethod
    def _calculate_grade(load_time_ms: float) -> str:
        """Calculate performance grade"""
        if load_time_ms < 1000:
            return "A+"
        elif load_time_ms < 2000:
            return "A"
        elif load_time_ms < 3000:
            return "B"
        elif load_time_ms < 5000:
            return "C"
        elif load_time_ms < 8000:
            return "D"
        else:
            return "F"
    
    @staticmethod
    async def analyze_images(soup: BeautifulSoup, base_url: str) -> Dict:
        """Analyze images for optimization"""
        images = soup.find_all('img')
        
        results = {
            "total": len(images),
            "missing_alt": 0,
            "missing_title": 0,
            "missing_dimensions": 0,
            "lazy_loading": 0,
            "large_images": [],
            "external_images": 0
        }
        
        for img in images:
            # Check alt text
            if not img.get('alt'):
                results['missing_alt'] += 1
            
            # Check title
            if not img.get('title'):
                results['missing_title'] += 1
            
            # Check dimensions
            if not (img.get('width') and img.get('height')):
                results['missing_dimensions'] += 1
            
            # Check lazy loading
            if img.get('loading') == 'lazy':
                results['lazy_loading'] += 1
            
            # Check if external
            src = img.get('src', '')
            if src.startswith(('http://', 'https://')):
                if not src.startswith(base_url):
                    results['external_images'] += 1
        
        return results


class StructuredDataAnalyzer:
    """Analyze structured data and schema markup"""
    
    @staticmethod
    def analyze(soup: BeautifulSoup) -> Dict:
        """Analyze structured data"""
        results = {
            "json_ld": [],
            "microdata": False,
            "rdfa": False,
            "opengraph": {},
            "twitter_cards": {},
            "total_schemas": 0
        }
        
        # JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                schema_type = data.get('@type', 'Unknown')
                results['json_ld'].append(schema_type)
            except:
                pass
        
        # Microdata
        if soup.find(attrs={'itemscope': True}):
            results['microdata'] = True
        
        # RDFa
        if soup.find(attrs={'vocab': True}) or soup.find(attrs={'typeof': True}):
            results['rdfa'] = True
        
        # Open Graph
        og_tags = soup.find_all('meta', property=re.compile('^og:'))
        for tag in og_tags:
            prop = tag.get('property', '').replace('og:', '')
            results['opengraph'][prop] = tag.get('content', '')
        
        # Twitter Cards
        twitter_tags = soup.find_all('meta', attrs={'name': re.compile('^twitter:')})
        for tag in twitter_tags:
            prop = tag.get('name', '').replace('twitter:', '')
            results['twitter_cards'][prop] = tag.get('content', '')
        
        results['total_schemas'] = len(results['json_ld']) + \
                                   (1 if results['microdata'] else 0) + \
                                   (1 if results['rdfa'] else 0)
        
        return results


class LighthouseAnalyzer:
    """Google PageSpeed Insights / Lighthouse integration"""
    
    @staticmethod
    async def analyze(url: str, strategy: str = "mobile") -> Dict:
        """Get Lighthouse metrics from PageSpeed API"""
        if not PAGESPEED_API_KEY:
            return {"error": "PageSpeed API key not configured"}
            
        try:
            params = {
                "url": url,
                "key": PAGESPEED_API_KEY,
                "strategy": strategy,
                "category": ["performance", "accessibility", "best-practices", "seo"]
            }
            
            response = requests.get(PAGESPEED_API_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}"}
            
            data = response.json()
            lighthouse = data.get("lighthouseResult", {})
            
            # Extract categories
            categories = lighthouse.get("categories", {})
            
            # Extract audits
            audits = lighthouse.get("audits", {})
            
            # Core Web Vitals
            metrics = {
                "scores": {
                    "performance": int(categories.get("performance", {}).get("score", 0) * 100),
                    "accessibility": int(categories.get("accessibility", {}).get("score", 0) * 100),
                    "best_practices": int(categories.get("best-practices", {}).get("score", 0) * 100),
                    "seo": int(categories.get("seo", {}).get("score", 0) * 100)
                },
                "metrics": {
                    "fcp": audits.get("first-contentful-paint", {}).get("displayValue", "N/A"),
                    "lcp": audits.get("largest-contentful-paint", {}).get("displayValue", "N/A"),
                    "cls": audits.get("cumulative-layout-shift", {}).get("displayValue", "N/A"),
                    "tti": audits.get("interactive", {}).get("displayValue", "N/A"),
                    "tbt": audits.get("total-blocking-time", {}).get("displayValue", "N/A"),
                    "speed_index": audits.get("speed-index", {}).get("displayValue", "N/A")
                },
                "opportunities": [],
                "diagnostics": []
            }
            
            # Extract opportunities
            for key, audit in audits.items():
                if audit.get("score", 1) < 1:
                    if "details" in audit and "overallSavingsMs" in audit.get("details", {}):
                        metrics["opportunities"].append({
                            "title": audit.get("title"),
                            "savings": audit.get("displayValue")
                        })
            
            return metrics
            
        except Exception as e:
            return {"error": str(e)}


# =======================
# MAIN SEO ANALYZER
# =======================

class SEOAnalyzer:
    """Main SEO analysis orchestrator"""
    
    def __init__(self):
        self.issues = {
            "errors": [],
            "warnings": [],
            "notices": []
        }
        self.metrics = {}
        self.issue_definitions = SEOIssueDefinitions()
    
    def clear_issues(self):
        """Clear all issues"""
        self.issues = {
            "errors": [],
            "warnings": [],
            "notices": []
        }
        self.metrics = {}
    
    def add_issue(self, code: int, details: str = "", affected_elements: List = None):
        """Add an issue based on code"""
        issue_info = self.issue_definitions.get_issue_details(code)
        
        issue = {
            "code": code,
            "title": issue_info["title"],
            "severity": issue_info["severity"],
            "category": issue_info["category"],
            "details": details,
            "affected_elements": affected_elements or []
        }
        
        # Add to appropriate list
        if issue_info["severity"] == "error":
            self.issues["errors"].append(issue)
        elif issue_info["severity"] == "warning":
            self.issues["warnings"].append(issue)
        else:
            self.issues["notices"].append(issue)
    
    async def analyze(self, url: str) -> Dict:
        """Perform complete SEO analysis"""
        self.clear_issues()
        
        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Store base URL
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Initialize results
        results = {
            "url": url,
            "base_url": base_url,
            "timestamp": datetime.now().isoformat(),
            "crawler_used": "requests"
        }
        
        try:
            # Use requests
            response = requests.get(url, timeout=10, verify=False)
            html_content = response.text
            results["status_code"] = response.status_code
            results["response_headers"] = dict(response.headers)
            self._check_http_status(response.status_code)
            
            # Parse HTML with html5lib (no lxml needed)
            soup = BeautifulSoup(html_content, 'html5lib')
            
            # Basic checks
            await self._check_meta_tags(soup)
            await self._check_headings(soup)
            await self._check_links(soup, base_url)
            await self._check_images(soup)
            await self._check_viewport(soup)
            await self._check_content(soup)
            await self._check_technical_seo(soup, html_content)
            
            # Advanced checks
            results["robots_txt"] = await RobotsTxtParser.parse(base_url)
            results["sitemap"] = await SitemapParser.parse(base_url)
            results["ssl"] = await SecurityChecker.check_ssl(url)
            
            # Security headers
            results["security_headers"] = await SecurityChecker.check_security_headers(
                results["response_headers"]
            )
            
            # Content analysis
            clean_text = ContentAnalyzer.extract_clean_text(soup)
            results["content"] = {
                "word_count": len(clean_text.split()),
                "readability": ContentAnalyzer.analyze_readability(clean_text),
                "keywords": ContentAnalyzer.analyze_keywords(clean_text, soup),
                "sentiment": ContentAnalyzer.analyze_sentiment(clean_text)
            }
            
            # Performance
            results["performance"] = await PerformanceAnalyzer.measure_load_time(url)
            results["images_analysis"] = await PerformanceAnalyzer.analyze_images(soup, base_url)
            
            # Structured data
            results["structured_data"] = StructuredDataAnalyzer.analyze(soup)
            
            # Lighthouse (if API key is valid)
            if PAGESPEED_API_KEY:
                results["lighthouse"] = await LighthouseAnalyzer.analyze(url)
            
            # Additional checks based on results
            self._check_robots_txt(results.get("robots_txt", {}))
            self._check_sitemap(results.get("sitemap", {}))
            self._check_ssl(results.get("ssl", {}))
            
            # Compile final results
            results["issues"] = self.issues
            results["total_issues"] = {
                "errors": len(self.issues["errors"]),
                "warnings": len(self.issues["warnings"]),
                "notices": len(self.issues["notices"]),
                "total": len(self.issues["errors"]) + len(self.issues["warnings"]) + len(self.issues["notices"])
            }
            
            # Calculate SEO score
            results["seo_score"] = self._calculate_seo_score()
            
            return results
            
        except requests.RequestException as e:
            self.add_issue(11, f"Failed to fetch URL: {str(e)}")
            results["error"] = str(e)
            results["issues"] = self.issues
            return results
        
        except Exception as e:
            results["error"] = f"Analysis error: {str(e)}"
            results["issues"] = self.issues
            return results
    
    def _check_http_status(self, status_code: int):
        """Check HTTP status codes"""
        if status_code >= 500:
            self.add_issue(1, f"Server returned {status_code} error")
        elif status_code >= 400:
            self.add_issue(2, f"Page returned {status_code} error")
    
    async def _check_meta_tags(self, soup: BeautifulSoup):
        """Check meta tags"""
        # Title
        title = soup.find('title')
        if not title or not title.text.strip():
            self.add_issue(3)
        elif title:
            title_text = title.text.strip()
            title_length = len(title_text)
            
            if title_length < 30:
                self.add_issue(101, f"Title has only {title_length} characters")
            elif title_length > 60:
                self.add_issue(102, f"Title has {title_length} characters")
            
            # Store for duplicate checking
            self.metrics["title"] = title_text
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if not meta_desc or not meta_desc.get('content', '').strip():
            self.add_issue(106)
        elif meta_desc:
            desc_text = meta_desc.get('content', '').strip()
            desc_length = len(desc_text)
            
            if desc_length < 120:
                self.add_issue(106, f"Description has only {desc_length} characters")
            elif desc_length > 160:
                self.add_issue(106, f"Description has {desc_length} characters")
            
            self.metrics["meta_description"] = desc_text
        
        # Charset
        charset = soup.find('meta', charset=True)
        if not charset:
            charset = soup.find('meta', attrs={'http-equiv': 'Content-Type'})
        
        if not charset:
            self.add_issue(115)
        
        # Canonical
        canonical = soup.find('link', rel='canonical')
        if canonical:
            canonical_url = canonical.get('href')
            if not canonical_url:
                self.add_issue(38)
        
        # Check for multiple canonicals
        canonicals = soup.find_all('link', rel='canonical')
        if len(canonicals) > 1:
            self.add_issue(39, f"Found {len(canonicals)} canonical URLs")
    
    async def _check_headings(self, soup: BeautifulSoup):
        """Check heading structure"""
        h1_tags = soup.find_all('h1')
        
        if len(h1_tags) == 0:
            self.add_issue(103)
        elif len(h1_tags) > 1:
            self.add_issue(104, f"Found {len(h1_tags)} H1 tags")
        
        # Check if H1 matches title
        if h1_tags and self.metrics.get("title"):
            h1_text = h1_tags[0].text.strip()
            if h1_text.lower() == self.metrics["title"].lower():
                self.add_issue(105)
    
    async def _check_links(self, soup: BeautifulSoup, base_url: str):
        """Check internal and external links"""
        links = soup.find_all('a', href=True)
        
        internal_links = []
        external_links = []
        broken_links = []
        nofollow_internal = []
        
        # Group issues by type
        empty_anchor_links = []
        non_descriptive_anchors = []
        underscore_urls = []
        long_urls = []
        malformed_links_list = []
        http_from_https_links = []
        
        for link in links:
            href = link.get('href', '')
            
            # Skip special links
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Normalize URL
            if not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(base_url, href)
            
            # Parse URL
            try:
                parsed = urllib.parse.urlparse(href)
                
                # Check if internal or external
                if parsed.netloc == urllib.parse.urlparse(base_url).netloc:
                    internal_links.append(href)
                    
                    # Check for nofollow on internal links
                    if link.get('rel') and 'nofollow' in link.get('rel'):
                        nofollow_internal.append(href)
                else:
                    external_links.append(href)
                    
                    # Check if linking to HTTP from HTTPS
                    if base_url.startswith('https') and href.startswith('http:'):
                        http_from_https_links.append(href[:100])
                
                # Check for underscores in URL
                if '_' in parsed.path:
                    underscore_urls.append(href[:100])
                
                # Check URL length
                if len(href) > 200:
                    long_urls.append(href[:100])
                
                # Check anchor text
                anchor_text = link.text.strip()
                if not anchor_text:
                    empty_anchor_links.append(href[:100])
                elif anchor_text.lower() in ['click here', 'read more', 'link', 'here']:
                    non_descriptive_anchors.append({
                        'text': anchor_text,
                        'url': href[:100]
                    })
                    
            except Exception:
                malformed_links_list.append(href[:50])
        
        # Add grouped issues
        if empty_anchor_links:
            self.add_issue(216, f"Found {len(empty_anchor_links)} links with empty anchor text", empty_anchor_links)
        
        if non_descriptive_anchors:
            self.add_issue(217, f"Found {len(non_descriptive_anchors)} links with non-descriptive anchor text", non_descriptive_anchors)
        
        if underscore_urls:
            self.add_issue(122, f"Found {len(underscore_urls)} URLs containing underscores", underscore_urls)
        
        if long_urls:
            self.add_issue(201, f"Found {len(long_urls)} URLs that are too long (>200 chars)", long_urls)
        
        if malformed_links_list:
            self.add_issue(44, f"Found {len(malformed_links_list)} malformed links", malformed_links_list)
        
        if http_from_https_links:
            self.add_issue(31, f"Found {len(http_from_https_links)} insecure HTTP links on HTTPS site", http_from_https_links)
        
        # Check total links
        total_links = len(internal_links) + len(external_links)
        if total_links > 100:
            self.add_issue(108, f"Page has {total_links} total links (recommended: under 100)")
        
        # Check for nofollow on internal links
        if nofollow_internal:
            self.add_issue(123, f"Found {len(nofollow_internal)} internal links with nofollow", nofollow_internal)
        
        self.metrics["internal_links"] = len(internal_links)
        self.metrics["external_links"] = len(external_links)
    
    async def _check_images(self, soup: BeautifulSoup):
        """Check images"""
        images = soup.find_all('img')
        
        missing_alt = []
        missing_dimensions = []
        
        for img in images:
            # Check alt text
            if not img.get('alt'):
                src = img.get('src', 'unknown')[:100]
                missing_alt.append(src)
            
            # Check dimensions
            if not (img.get('width') and img.get('height')):
                src = img.get('src', 'unknown')[:100]
                missing_dimensions.append(src)
        
        # Add grouped issues
        if missing_alt:
            self.add_issue(110, f"Found {len(missing_alt)} images without alt text", missing_alt)
        
        if missing_dimensions:
            self.add_issue(110, f"Found {len(missing_dimensions)} images without dimensions specified", missing_dimensions)
    
    async def _check_viewport(self, soup: BeautifulSoup):
        """Check viewport and mobile optimization"""
        viewport = soup.find('meta', attrs={'name': 'viewport'})
        
        if not viewport:
            self.add_issue(20)
        elif viewport:
            content = viewport.get('content', '')
            if 'width' not in content:
                self.add_issue(46)
    
    async def _check_content(self, soup: BeautifulSoup):
        """Check content quality"""
        text = ContentAnalyzer.extract_clean_text(soup)
        word_count = len(text.split())
        
        if word_count < 300:
            self.add_issue(117, f"Only {word_count} words found")
        elif word_count > 10000:
            self.add_issue(220, f"Very long content: {word_count} words")
        
        # Check text to HTML ratio
        html_size = len(str(soup))
        text_size = len(text)
        
        if html_size > 0:
            text_ratio = (text_size / html_size) * 100
            if text_ratio < 10:
                self.add_issue(112, f"Text ratio is only {text_ratio:.1f}%")
    
    async def _check_technical_seo(self, soup: BeautifulSoup, html_content: str):
        """Check technical SEO aspects"""
        # Doctype
        if not html_content.lower().startswith('<!doctype'):
            self.add_issue(116)
        
        # Check for frames
        if soup.find('frame') or soup.find('frameset'):
            self.add_issue(121)
        
        # Check for meta refresh
        meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
        if meta_refresh:
            self.add_issue(40)
        
        # Check page size
        page_size_kb = len(html_content) / 1024
        if page_size_kb > 200:
            self.add_issue(21, f"HTML size is {page_size_kb:.1f} KB")
        
        # Check for inline styles and scripts
        inline_styles = len(soup.find_all('style'))
        inline_scripts = len(soup.find_all('script', src=False))
        
        if inline_styles > 5:
            self.add_issue(134, f"Too many inline styles: {inline_styles}")
        
        if inline_scripts > 5:
            self.add_issue(134, f"Too many inline scripts: {inline_scripts}")
    
    def _check_robots_txt(self, robots_data: dict):
        """Check robots.txt issues"""
        if not robots_data.get("exists"):
            self.add_issue(203)
        elif not robots_data.get("valid"):
            self.add_issue(16, robots_data.get("error", "Invalid format"))
        else:
            # Check if sitemap is specified
            if not robots_data.get("sitemaps"):
                self.add_issue(124)
    
    def _check_sitemap(self, sitemap_data: dict):
        """Check sitemap issues"""
        if not sitemap_data.get("exists"):
            self.add_issue(125)
        elif not sitemap_data.get("valid"):
            self.add_issue(17, sitemap_data.get("error", "Invalid format"))
        elif sitemap_data.get("too_large"):
            self.add_issue(43, f"Sitemap is {sitemap_data.get('size_mb', 0)} MB")
    
    def _check_ssl(self, ssl_data: dict):
        """Check SSL/HTTPS issues"""
        if not ssl_data.get("https"):
            self.add_issue(126)
        elif ssl_data.get("expired"):
            self.add_issue(27, "Certificate has expired")
        elif ssl_data.get("expiring_soon"):
            self.add_issue(27, f"Certificate expires in {ssl_data.get('expires_in_days')} days")
        
        if ssl_data.get("old_protocol"):
            self.add_issue(28, f"Using {ssl_data.get('protocol_version')}")
        
        if ssl_data.get("https") and not ssl_data.get("hostname_match"):
            self.add_issue(29)
    
    def _calculate_seo_score(self) -> int:
        """Calculate overall SEO score"""
        # Start with 100
        score = 100
        
        # Deduct for issues
        score -= len(self.issues["errors"]) * 10
        score -= len(self.issues["warnings"]) * 5
        score -= len(self.issues["notices"]) * 2
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))


# =======================
# API ENDPOINTS
# =======================

class AnalysisRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page with simple form"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SEOMetrics Professional</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
            }
            h1 { color: #2c3e50; }
            .form-group {
                margin: 20px 0;
            }
            input[type="text"] {
                width: 100%;
                padding: 10px;
                font-size: 16px;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            button {
                background: #3498db;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background: #2980b9;
            }
        </style>
    </head>
    <body>
        <h1>🔍 SEOMetrics Professional</h1>
        <p>Comprehensive SEO analysis tool</p>
        
        <div class="form-group">
            <input type="text" id="url" placeholder="Enter website URL (e.g., example.com)" />
        </div>
        
        <button onclick="analyze()">Analyze Website</button>
        
        <div id="result"></div>
        
        <script>
            function analyze() {
                const url = document.getElementById('url').value;
                if (!url) {
                    alert('Please enter a URL');
                    return;
                }
                
                document.getElementById('result').innerHTML = '<p>Analyzing... Please wait...</p>';
                
                fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: url })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerHTML = 
                        '<h2>Analysis Complete!</h2>' +
                        '<p>SEO Score: ' + data.seo_score + '/100</p>' +
                        '<p>Errors: ' + data.total_issues.errors + '</p>' +
                        '<p>Warnings: ' + data.total_issues.warnings + '</p>' +
                        '<p><a href="/api/analyze?url=' + encodeURIComponent(url) + '" target="_blank">View Full Report</a></p>';
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = '<p style="color:red;">Error: ' + error + '</p>';
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/api/analyze")
async def analyze_post(request: AnalysisRequest):
    """Analyze a website - POST endpoint"""
    analyzer = SEOAnalyzer()
    results = await analyzer.analyze(request.url)
    return results

@app.get("/api/analyze")
async def analyze_get(url: str):
    """Analyze a website - GET endpoint"""
    analyzer = SEOAnalyzer()
    results = await analyzer.analyze(url)
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "groq_available": GROQ_AVAILABLE,
        "pagespeed_configured": bool(PAGESPEED_API_KEY)
    }


# =======================
# SERVER STARTUP
# =======================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
