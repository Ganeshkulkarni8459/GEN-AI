// types.ts - Type definitions for the Multi-Modal App

export interface Result {
  intent: string;
  type: 'text' | 'image' | 'audio' | 'translation';
  result: string;
  original?: string;
  file_path?: string;
  image_data?: string;
  audio_data?: string;
}

export interface Example {
  label: string;
  text: string;
}

export interface ApiResponse {
  intent: string;
  type: string;
  result: string;
  original?: string;
  file_path?: string;
  image_data?: string;
  audio_data?: string;
}

export interface ApiErrorResponse {
  detail: string;
}

export interface ProcessRequest {
  prompt: string;
}