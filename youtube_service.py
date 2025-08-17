import yt_dlp
import re
import logging
from urllib.parse import urlparse, parse_qs

def extract_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(youtube_url)
    
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    
    # Try to extract from URL using regex as fallback
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    if match:
        return match.group(1)
    
    raise ValueError("Invalid YouTube URL format")

def extract_youtube_transcript(youtube_url):
    """
    Extract transcript from YouTube video using yt-dlp
    Returns dict with video_id, title, and transcript
    """
    try:
        video_id = extract_video_id(youtube_url)
        
        # Configure yt-dlp options
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info
            info = ydl.extract_info(youtube_url, download=False)
            
            title = info.get('title', 'Unknown Title')
            
            # Try to get subtitles/transcript
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            
            transcript_text = ""
            
            # Try manual subtitles first (more accurate)
            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subtitles:
                    # Get the subtitle URL and download
                    subtitle_info = subtitles[lang]
                    if subtitle_info:
                        # Find the best format (prefer vtt or srv3)
                        best_format = None
                        for fmt in subtitle_info:
                            if fmt.get('ext') in ['vtt', 'srv3']:
                                best_format = fmt
                                break
                        
                        if not best_format and subtitle_info:
                            best_format = subtitle_info[0]
                        
                        if best_format:
                            try:
                                subtitle_url = best_format['url']
                                transcript_text = _download_and_parse_subtitles(subtitle_url, best_format.get('ext', 'vtt'))
                                break
                            except Exception as e:
                                logging.warning(f"Error downloading manual subtitles: {e}")
                                continue
            
            # If no manual subtitles, try automatic captions
            if not transcript_text:
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in automatic_captions:
                        caption_info = automatic_captions[lang]
                        if caption_info:
                            # Find the best format
                            best_format = None
                            for fmt in caption_info:
                                if fmt.get('ext') in ['vtt', 'srv3']:
                                    best_format = fmt
                                    break
                            
                            if not best_format and caption_info:
                                best_format = caption_info[0]
                            
                            if best_format:
                                try:
                                    caption_url = best_format['url']
                                    transcript_text = _download_and_parse_subtitles(caption_url, best_format.get('ext', 'vtt'))
                                    break
                                except Exception as e:
                                    logging.warning(f"Error downloading automatic captions: {e}")
                                    continue
            
            if not transcript_text:
                raise Exception("No English transcript/captions available for this video")
            
            return {
                'video_id': video_id,
                'title': title,
                'transcript': transcript_text
            }
            
    except Exception as e:
        logging.error(f"Error extracting YouTube transcript: {e}")
        raise Exception(f"Failed to extract transcript: {str(e)}")

def _download_and_parse_subtitles(url, format_type):
    """Download and parse subtitle file"""
    import requests
    import xml.etree.ElementTree as ET
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        if format_type == 'srv3' or 'srv3' in url:
            # XML format (srv3)
            return _parse_xml_subtitles(content)
        else:
            # VTT format
            return _parse_vtt_subtitles(content)
            
    except Exception as e:
        logging.error(f"Error downloading subtitles from {url}: {e}")
        raise

def _parse_xml_subtitles(xml_content):
    """Parse XML subtitle format (srv3)"""
    try:
        root = ET.fromstring(xml_content)
        transcript_parts = []
        
        for text_element in root.findall('.//text'):
            text_content = text_element.text
            if text_content:
                # Clean up the text
                text_content = re.sub(r'<[^>]+>', '', text_content)  # Remove HTML tags
                text_content = text_content.strip()
                if text_content:
                    transcript_parts.append(text_content)
        
        return ' '.join(transcript_parts)
        
    except Exception as e:
        logging.error(f"Error parsing XML subtitles: {e}")
        raise

def _parse_vtt_subtitles(vtt_content):
    """Parse VTT subtitle format"""
    try:
        lines = vtt_content.split('\n')
        transcript_parts = []
        
        for line in lines:
            line = line.strip()
            # Skip VTT headers, timestamps, and empty lines
            if (line.startswith('WEBVTT') or 
                line.startswith('NOTE') or 
                '-->' in line or 
                not line or
                line.isdigit()):
                continue
            
            # Clean up the text
            line = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
            line = re.sub(r'\{[^}]+\}', '', line)  # Remove style tags
            line = line.strip()
            
            if line:
                transcript_parts.append(line)
        
        return ' '.join(transcript_parts)
        
    except Exception as e:
        logging.error(f"Error parsing VTT subtitles: {e}")
        raise
