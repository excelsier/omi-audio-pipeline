#!/usr/bin/env python3
"""
Robust Direct PyAnnote Diarizer
-------------------------------
Enhanced version of the PyAnnote diarizer with improved reliability for large files.
Provides direct audio file processing without chunking to avoid speaker over-segmentation.

Key features:
- Processes entire audio file at once to maintain speaker consistency
- Enhanced job polling mechanism with proper status handling
- Robust error recovery and fallback mechanisms
- Configurable speaker clustering parameters
- Google Cloud Storage integration for large file handling
"""
import os
import time
import json
import uuid
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for PyAnnote API
PYANNOTE_API_URL = "https://api.pyannote.audio"
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")

# Default paths
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class OptimizedGCSManager:
    """Google Cloud Storage manager with optimized upload and URL signing"""
    
    def __init__(self, bucket_name=None, credentials_path=None):
        """
        Initialize GCS manager
        
        Args:
            bucket_name: GCS bucket name
            credentials_path: Path to GCS credentials JSON file
        """
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME", "omi-audio-streaming-files")
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Initialize client
        try:
            from google.cloud import storage
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"Optimized GCS client initialized for bucket: {self.bucket_name}")
        except ImportError:
            logger.warning("google-cloud-storage not installed, fallback to local processing")
            self.client = None
            self.bucket = None
    
    def upload_file(self, local_path, remote_path=None):
        """
        Upload file to GCS bucket with optional path
        
        Args:
            local_path: Path to local file
            remote_path: Optional remote path (default: filename)
            
        Returns:
            Remote blob name if successful, None otherwise
        """
        if not self.client:
            logger.error("GCS client not available")
            return None
        
        start_time = time.time()
        try:
            local_path = Path(local_path)
            remote_path = remote_path or f"diarization_input/{local_path.name}"
            
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(str(local_path))
            
            file_size = local_path.stat().st_size / (1024 * 1024)  # Size in MB
            elapsed = time.time() - start_time
            logger.info(f"File {local_path} uploaded to {remote_path} in {elapsed:.2f}s ({file_size:.2f}MB)")
            return remote_path
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return None
    
    def generate_signed_url(self, blob_name, expiration=86400):
        """
        Generate signed URL for GCS blob
        
        Args:
            blob_name: Name of the blob
            expiration: Expiration time in seconds (default: 24 hours)
            
        Returns:
            Signed URL if successful, None otherwise
        """
        if not self.client:
            logger.error("GCS client not available")
            return None
        
        try:
            blob = self.bucket.blob(blob_name)
            url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET"
            )
            logger.info(f"Generated signed URL for: {blob_name}")
            return url
        except Exception as e:
            logger.error(f"Error generating signed URL: {str(e)}")
            return None
    
    def delete_blob(self, blob_name):
        """
        Delete blob from GCS
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("GCS client not available")
            return False
        
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting blob: {str(e)}")
            return False


class RobustDirectPyAnnoteDiarizer:
    """Enhanced non-parallel PyAnnote diarizer with improved reliability for large files"""
    
    def __init__(self, api_key=None, polling_interval=5, gcs_manager=None,
                min_speakers=2, max_speakers=8, clustering_threshold=0.75,
                max_retries=3, extended_timeout=True):
        """
        Initialize enhanced PyAnnote Cloud API diarizer
        
        Args:
            api_key: PyAnnote API key
            polling_interval: Seconds between polling requests for job status
            gcs_manager: GCS Manager instance
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            clustering_threshold: Threshold for speaker clustering
            max_retries: Maximum number of API retry attempts
            extended_timeout: Whether to use extended timeout for large files
        """
        self.api_key = api_key or PYANNOTE_API_KEY
        self.polling_interval = polling_interval
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Speaker diarization parameters
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.clustering_threshold = clustering_threshold
        
        # Reliability parameters
        self.max_retries = max_retries
        self.extended_timeout = extended_timeout
        
        # Initialize optimized GCS manager if not provided
        if gcs_manager is None:
            self.gcs_manager = OptimizedGCSManager()
        else:
            self.gcs_manager = gcs_manager
            
        logger.info(f"Initialized robust direct diarizer with extended_timeout={extended_timeout}")
        logger.info(f"Speaker constraints: min={min_speakers}, max={max_speakers}, threshold={clustering_threshold}")
    
    def check_job_status(self, job_id):
        """
        Check the status of a diarization job with enhanced error handling
        
        Args:
            job_id: PyAnnote job ID
            
        Returns:
            Job status information
        """
        url = f"{PYANNOTE_API_URL}/v1/jobs/{job_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                status_data = response.json()
                
                # Handle different status field naming
                status = status_data.get("status", None)
                
                # API might use "succeeded", "completed", "processing", "running", "created" as status values
                if status in ("succeeded", "completed"):
                    return {"status": "completed", "result": status_data.get("result", {})}
                elif status == "failed":
                    return {"status": "failed", "error": status_data.get("error", "Unknown error")}
                elif status in ("processing", "running"):
                    return {"status": "processing"}
                elif status == "created":
                    # "created" means job was accepted but not started processing yet
                    return {"status": "processing", "message": "Job created but not yet processing"}
                else:
                    logger.warning(f"Unknown job status: {status}")
                    return {"status": "unknown", "raw_data": status_data}
                    
            elif response.status_code == 404:
                logger.error(f"Job not found: {job_id}")
                return {"status": "error", "error": "Job not found"}
                
            else:
                logger.error(f"Error checking job status: {response.status_code} - {response.text}")
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Exception checking job status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def wait_for_job_completion(self, job_id, timeout=900):
        """
        Wait for a job to complete with progressive polling
        
        Args:
            job_id: PyAnnote job ID
            timeout: Maximum wait time in seconds (default: 15 minutes)
            
        Returns:
            Job result or None if timeout/error
        """
        # Handle synchronous responses (no job polling needed)
        if isinstance(job_id, dict):
            # Check if this is a complete result or just a job status
            if "job_id" not in job_id and ("annotation" in job_id or "result" in job_id):
                logger.info("Using synchronous result (skipping polling)")
                return job_id
            # If it has a job_id field, extract it and poll that instead
            elif "jobId" in job_id:
                logger.info(f"Extracting jobId from response: {job_id['jobId']}")
                job_id = job_id['jobId']
            
        start_time = time.time()
        attempt = 0
        
        # Use a progressive polling interval to reduce API load
        base_interval = self.polling_interval
        
        logger.info(f"Waiting for job completion: {job_id} (timeout: {timeout}s)")
        
        while True:
            # Check if we've exceeded the timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Timeout waiting for job completion: {job_id}")
                return None
            
            # Calculate current polling interval with progressive backoff
            # Start with frequent checks, then gradually extend the interval
            current_interval = min(60, base_interval * (1.2 ** min(10, attempt)))
            
            # Check job status
            status_response = self.check_job_status(job_id)
            
            if status_response["status"] == "completed":
                logger.info(f"Job completed successfully after {elapsed:.1f}s: {job_id}")
                return status_response["result"]
            elif status_response["status"] == "failed":
                logger.error(f"Job failed after {elapsed:.1f}s: {job_id}")
                logger.error(f"Error details: {status_response.get('error', 'Unknown error')}")
                return None
            elif status_response["status"] == "processing":
                # Job is still processing, log and continue polling
                if "message" in status_response:
                    logger.info(f"Job {job_id} still processing... ({elapsed:.1f}s elapsed, {timeout-elapsed:.1f}s remaining, next check in {current_interval:.1f}s)")
                else:
                    logger.info(f"Job {job_id} still processing... ({elapsed:.1f}s elapsed, {timeout-elapsed:.1f}s remaining, next check in {current_interval:.1f}s)")
            elif status_response["status"] == "error":
                logger.error(f"Error checking job: {status_response.get('error', 'Unknown error')}")
                
                # Only consider the job lost after multiple error attempts
                if attempt >= 3:
                    logger.error(f"Too many errors checking job status, giving up")
                    return None
            
            # Increment attempt counter and sleep before next check
            attempt += 1
            time.sleep(current_interval)
    
    def submit_direct_upload(self, audio_path):
        """
        Submit audio directly to PyAnnote API without using a URL
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Job ID if successful, None otherwise
        """
        # Maximum attempts for API submission
        max_api_attempts = self.max_retries
        
        # Upload file to GCS for processing
        try:
            audio_path = Path(audio_path)
            blob_name = self.gcs_manager.upload_file(audio_path)
            
            if not blob_name:
                logger.error("Failed to upload audio to GCS")
                return None
            
            # Generate signed URL for the uploaded file
            signed_url = self.gcs_manager.generate_signed_url(blob_name)
            if not signed_url:
                logger.error("Failed to generate signed URL")
                return None
            
            logger.info(f"Submitting audio URL to PyAnnote API: {signed_url}")
        except Exception as e:
            logger.error(f"Error preparing audio for submission: {str(e)}")
            return None
        
        # Parameters for diarization
        params = {}
        
        # Only include parameters if they differ from PyAnnote defaults
        # Check if any non-default parameters are set
        if (self.min_speakers != 1 or 
            self.max_speakers != 20 or 
            self.clustering_threshold != 0.75):
            # Add clustering parameters
            params = {
                "min_speakers": self.min_speakers,
                "max_speakers": self.max_speakers,
                "clustering": "AgglomerativeClustering",
                "clustering_threshold": self.clustering_threshold
            }
        
        if params:
            logger.info(f"Using custom PyAnnote parameters: {params}")
        else:
            logger.info(f"Note: Using default PyAnnote parameters instead of: min_speakers={self.min_speakers}, max_speakers={self.max_speakers}, threshold={self.clustering_threshold}")
        
        # Try to submit the job with retry logic
        job_id = None
        response_data = None
        
        for attempt in range(max_api_attempts):
            try:
                url = f"{PYANNOTE_API_URL}/v1/diarize"
                data = {"url": signed_url}
                
                # Add parameters if specified
                if params:
                    data["params"] = params
                
                response = requests.post(url, headers=self.headers, json=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    job_id = response_data.get("jobId")
                    
                    if job_id:
                        logger.info(f"Diarization job submitted successfully: {job_id}")
                        break
                    else:
                        logger.info("Received synchronous response (no job_id)")
                        # This might be a synchronous result with no job_id
                        return response_data
                        
                elif response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited by PyAnnote API, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"Error submitting job: {response.status_code} - {response.text}")
                    if attempt < max_api_attempts - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s (attempt {attempt+1}/{max_api_attempts})")
                        time.sleep(wait_time)
                    else:
                        logger.error("All submission attempts failed")
                        return None
                        
            except Exception as e:
                logger.error(f"Exception submitting diarization job: {str(e)}")
                if attempt < max_api_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s (attempt {attempt+1}/{max_api_attempts})")
                    time.sleep(wait_time)
                else:
                    logger.error("All submission attempts failed")
                    return None
        
        # Return either the job_id or response data
        return job_id or response_data
    
    def process_audio(self, audio_path, output_id=None):
        """
        Process a complete audio file with PyAnnote diarization
        
        Args:
            audio_path: Path to audio file
            output_id: Identifier for output files (default: timestamp)
            
        Returns:
            Diarization result
        """
        start_time = time.time()
        
        # Ensure audio path is a Path object
        audio_path = Path(audio_path)
        
        # Generate output ID if not provided
        if not output_id:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_id = f"{timestamp}_{audio_path.stem}"
        
        # Get audio metadata to determine appropriate timeout
        try:
            import librosa
            audio_duration = librosa.get_duration(filename=str(audio_path))
            logger.info(f"Processing audio: {audio_path} ({audio_duration:.1f}s)")
            
            # For large files, use extended timeout
            if audio_duration > 3600 and self.extended_timeout:
                logger.info(f"Large file detected ({audio_duration:.1f}s), enabling extended timeout")
        except Exception as e:
            logger.warning(f"Error getting audio duration: {str(e)}")
            audio_duration = 0
        
        # Submit the diarization job
        logger.info(f"Uploading audio to GCS: {audio_path}")
        job_result = self.submit_direct_upload(audio_path)
        
        # If submit_direct_upload returned None, create a fallback result
        if job_result is None:
            logger.error("Failed to submit diarization job, creating fallback result")
            return self._create_fallback_result(self.min_speakers)
        
        # Wait for job completion if we got a job ID
        if isinstance(job_result, str):
            # Calculate an appropriate timeout based on file duration
            # Minimum 15 minutes, plus 5 seconds per second of audio
            if self.extended_timeout and audio_duration > 0:
                timeout = max(900, int(audio_duration * 5))
            else:
                timeout = 900
            
            job_result = self.wait_for_job_completion(job_result, timeout=timeout)
            
            # If job_result is None (timeout or error), create a fallback result
            if job_result is None:
                logger.error("Job completion timed out or failed, creating fallback result")
                return self._create_fallback_result(self.min_speakers)
        
        # Clean up GCS blob if possible
        try:
            blob_name = f"diarization_input/{audio_path.name}"
            if not self.gcs_manager.delete_blob(blob_name):
                logger.warning(f"Could not delete GCS blob: {blob_name}")
        except Exception as e:
            logger.warning(f"Error deleting GCS blob {blob_name}: {str(e)}")
        
        # Format the result
        formatted_result = self._format_diarization_result(job_result, audio_path)
        
        # Save result to file
        output_path = RESULTS_DIR / f"{output_id}_diarization.json"
        with open(output_path, "w") as f:
            json.dump(formatted_result, f, indent=2)
        
        logger.info(f"Diarization complete. Results saved to {output_path}")
        logger.info(f"Detected {formatted_result.get('num_speakers', 0)} speakers")
        
        return formatted_result
    
    def _format_diarization_result(self, api_result, audio_path):
        """
        Format PyAnnote API result into a standardized structure
        
        Args:
            api_result: Raw API result
            audio_path: Path to source audio file
            
        Returns:
            Formatted result
        """
        logger.info(f"Formatting API result with keys: {list(api_result.keys())}")
        
        # Add detailed logging of the API response structure - helps with debugging
        try:
            import json
            detailed_str = json.dumps(api_result, indent=2)
            logger.debug(f"API Response structure:\n{detailed_str[:1000]}...")
        except Exception as e:
            logger.debug(f"Could not log detailed API response: {str(e)}")
        
        # Get audio info for duration and other metadata
        audio_info = None
        audio_duration = 0
        if isinstance(audio_path, (str, Path)):
            try:
                import librosa
                audio_duration = librosa.get_duration(filename=str(audio_path))
            except Exception as e:
                logger.warning(f"Error getting audio duration: {str(e)}")
        
        # First handle the case where we need to poll for the job result
        if "jobId" in api_result and "status" in api_result:
            job_id = api_result.get("jobId")
            status = api_result.get("status")
            
            # If job is created/processing/running, we need to poll for it
            if status in ("created", "processing", "running"):
                logger.info(f"Job {job_id} is in {status} state, polling until completion")
                # Set a generous timeout for processing (5 minutes minimum, 15 minutes for long files)
                timeout = max(300, int(audio_duration * 5))
                
                # Poll for job completion
                result = self.wait_for_job_completion(job_id, timeout=timeout)
                if result:
                    logger.info(f"Successfully retrieved result after polling")
                    # Update the API result with the completed job result
                    api_result = result
                    # Successfully retrieved result, no timeout occurred
                    return self._process_api_result(api_result, audio_path, audio_duration)
                else:
                    logger.warning(f"Polling for job {job_id} timed out after {timeout}s")
                    # Actually timed out, create fallback result
                    return self._create_fallback_result(self.min_speakers)
            
            elif status == "completed":
                # The job is completed, but we need to get the actual result from the endpoint
                logger.info(f"Job {job_id} completed, retrieving result")
                result_url = f"{PYANNOTE_API_URL}/v1/jobs/{job_id}/result"
                try:
                    response = requests.get(result_url, headers=self.headers)
                    if response.status_code == 200:
                        # Replace the job status with the actual result
                        api_result = response.json()
                        logger.info(f"Successfully retrieved result with keys: {list(api_result.keys())}")
                    else:
                        logger.error(f"Error getting job result: {response.status_code} - {response.text}")
                        return self._create_fallback_result(self.min_speakers)
                except Exception as e:
                    logger.error(f"Exception getting job result: {str(e)}")
                    return self._create_fallback_result(self.min_speakers)
            else:
                # This is not an error - just log it and continue with what we have
                # The job might be in some other state
                logger.info(f"Job status is '{status}', continuing with available data")
                
            # Check if we have usable data after all the processing
            if "annotation" in api_result or "segments" in api_result:
                logger.info(f"Found usable data in response")
                return self._process_api_result(api_result, audio_path, audio_duration)
            else:
                # Only create fallback if there's really no usable data
                logger.warning(f"No usable data found in job response, creating fallback")
                return self._create_fallback_result(self.min_speakers)
        
        # Process the result with our dedicated method
        return self._process_api_result(api_result, audio_path, audio_duration)
    
    def _process_api_result(self, api_result, audio_path, audio_duration):
        """
        Process API result data into a standardized format
        
        Args:
            api_result: Raw API result data
            audio_path: Path to source audio file
            audio_duration: Duration of the audio in seconds
            
        Returns:
            Formatted result dictionary
        """
        # -- Handle various API result formats --
        
        # Extract segments - try different formats based on the known response structures
        segments = []
        
        # Format used by DirectPyAnnoteDiarizer: "annotation" with "segments" 
        if "annotation" in api_result:
            annotation = api_result.get("annotation", {})
            raw_segments = annotation.get("segments", [])
            
            # This matches the working format in DirectPyAnnoteDiarizer
            if raw_segments:
                logger.info(f"Found {len(raw_segments)} segments in annotation format")
                for segment in raw_segments:
                    # Format each segment to our standard format
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    label = segment.get("label", "SPEAKER_0")
                    
                    # Extract speaker ID/number from the label
                    if label.startswith("SPEAKER_"):
                        try:
                            speaker_id = label
                        except ValueError:
                            speaker_id = f"SPEAKER_{len(segments)}"
                    else:
                        speaker_id = label
                    
                    segments.append({
                        "start_time": start,
                        "end_time": end,
                        "speaker": speaker_id,
                        "speaker_id": speaker_id,
                        "duration": end - start
                    })
        
        # Alternative format from recent PyAnnote API
        elif "segments" in api_result:
            raw_segments = api_result.get("segments", [])
            if raw_segments:
                logger.info(f"Found {len(raw_segments)} segments in direct format")
                for i, segment in enumerate(raw_segments):
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    speaker_id = segment.get("speaker", f"SPEAKER_{i+1}")
                    
                    segments.append({
                        "start_time": start,
                        "end_time": end,
                        "speaker": speaker_id,
                        "speaker_id": speaker_id,
                        "duration": end - start
                    })
        
        # If no segments were found in any format, create a fallback
        if not segments:
            logger.warning("No segments found in API response")
            return self._create_fallback_result(self.min_speakers)
        
        # Count unique speakers
        speakers = {}
        for segment in segments:
            speaker_id = segment["speaker_id"]
            if speaker_id not in speakers:
                speakers[speaker_id] = []
            speakers[speaker_id].append(segment)
        
        num_speakers = len(speakers)
        logger.info(f"Found {num_speakers} speakers")
        
        # Calculate speaking time for each speaker
        for speaker_id, speaker_segments in speakers.items():
            total_duration = sum(seg["duration"] for seg in speaker_segments)
            # Calculate percentage if we have audio duration
            if audio_duration > 0:
                speaking_percentage = (total_duration / audio_duration) * 100
            else:
                speaking_percentage = 0
            speakers[speaker_id] = {
                "segments": speaker_segments,
                "total_duration": total_duration,
                "speaking_percentage": speaking_percentage
            }
        
        # Create the standardized result
        file_name = Path(audio_path).name if isinstance(audio_path, (str, Path)) else "unknown"
        
        result = {
            "file_name": file_name,
            "num_speakers": num_speakers,
            "speakers": speakers,
            "segments": segments,
            "duration": audio_duration,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def _create_fallback_result(self, num_speakers=2):
        """
        Create a minimal fallback result
        
        Args:
            num_speakers: Number of speakers to include
            
        Returns:
            Fallback result structure
        """
        logger.warning(f"Creating fallback result with {num_speakers} speakers")
        
        segments = []
        speakers = {}
        
        # Create segments of equal length
        segment_duration = 5.0  # 5 seconds per segment
        
        for i in range(num_speakers):
            speaker_id = f"SPEAKER_{i+1}"
            start_time = i * segment_duration
            end_time = start_time + segment_duration
            
            segment = {
                "start_time": start_time,
                "end_time": end_time,
                "speaker": speaker_id,
                "speaker_id": speaker_id,
                "duration": segment_duration
            }
            
            segments.append(segment)
            
            speakers[speaker_id] = {
                "segments": [segment],
                "total_duration": segment_duration,
                "speaking_percentage": 0
            }
        
        # Create the standardized result
        result = {
            "file_name": "fallback_result",
            "num_speakers": num_speakers,
            "speakers": speakers,
            "segments": segments,
            "duration": num_speakers * segment_duration,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_fallback": True
        }
        
        return result


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio with PyAnnote cloud diarization")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--min-speakers", type=int, default=2, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=8, help="Maximum number of speakers")
    parser.add_argument("--threshold", type=float, default=0.75, help="Clustering threshold")
    
    args = parser.parse_args()
    
    diarizer = RobustDirectPyAnnoteDiarizer(
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        clustering_threshold=args.threshold
    )
    
    result = diarizer.process_audio(args.audio_path)
    print(f"Detected {result['num_speakers']} speakers")