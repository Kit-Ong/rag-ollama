'use client';

import { useState, useEffect } from 'react';
import { 
  X, 
  Database, 
  Upload, 
  Settings, 
  FileText, 
  Activity,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';
import axios from 'axios';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

interface StatusData {
  collection_exists: boolean;
  document_count: number;
  total_chunks: number;
  status_message: string;
}

interface DocumentData {
  name: string;
  path: string;
  size: number;
  modified: number;
}

const API_BASE_URL = 'http://localhost:8000';

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const [activeTab, setActiveTab] = useState<'status' | 'documents' | 'upload' | 'settings'>('status');
  const [status, setStatus] = useState<StatusData | null>(null);
  const [documents, setDocuments] = useState<DocumentData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetchStatus();
      fetchDocuments();
    }
  }, [isOpen]);

  const fetchStatus = async () => {
    try {
      const response = await axios.get<StatusData>(`${API_BASE_URL}/status`);
      setStatus(response.data);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`);
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleProcessDocuments = async () => {
    setIsLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/process-documents`);
      await fetchStatus();
    } catch (error) {
      console.error('Error processing documents:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!uploadFile) return;
    
    setUploadProgress(true);
    const formData = new FormData();
    formData.append('file', uploadFile);
    
    try {
      await axios.post(`${API_BASE_URL}/upload-document`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadFile(null);
      await fetchDocuments();
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setUploadProgress(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Overlay */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
        onClick={onClose}
      />
      
      {/* Sidebar */}
      <div className="fixed left-0 top-0 h-full w-80 bg-white shadow-xl z-50 lg:relative lg:translate-x-0 transform transition-transform">
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Control Panel</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-lg lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex">
            {[
              { id: 'status', label: 'Status', icon: Activity },
              { id: 'documents', label: 'Docs', icon: FileText },
              { id: 'upload', label: 'Upload', icon: Upload },
              { id: 'settings', label: 'Settings', icon: Settings },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`flex-1 flex flex-col items-center py-3 px-2 text-xs font-medium border-b-2 ${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <Icon className="w-4 h-4 mb-1" />
                {label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'status' && (
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-900 mb-3">Database Status</h3>
                {status ? (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      {status.collection_exists ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-yellow-500" />
                      )}
                      <span className="text-sm text-gray-600">
                        {status.collection_exists ? 'Connected' : 'Not Ready'}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      Documents: {status.document_count}
                    </div>
                    <div className="text-sm text-gray-600">
                      Chunks: {status.total_chunks}
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      {status.status_message}
                    </p>
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">Loading...</div>
                )}
              </div>

              <button
                onClick={handleProcessDocuments}
                disabled={isLoading}
                className="w-full bg-blue-600 text-white rounded-lg px-4 py-2 hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center space-x-2"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Database className="w-4 h-4" />
                )}
                <span>Process Documents</span>
              </button>
            </div>
          )}

          {activeTab === 'documents' && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-900">Documents ({documents.length})</h3>
              {documents.length > 0 ? (
                <div className="space-y-2">
                  {documents.map((doc, index) => (
                    <div key={index} className="bg-gray-50 rounded-lg p-3">
                      <div className="text-sm font-medium text-gray-900 truncate">
                        {doc.name}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {formatFileSize(doc.size)} • {formatDate(doc.modified)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-gray-500 text-center py-4">
                  No documents found
                </div>
              )}
            </div>
          )}

          {activeTab === 'upload' && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900">Upload Document</h3>
              
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept=".md,.txt,.pdf"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">
                    Click to select a file
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supports .md, .txt, .pdf files
                  </p>
                </label>
              </div>

              {uploadFile && (
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-sm font-medium text-gray-900">
                    {uploadFile.name}
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatFileSize(uploadFile.size)}
                  </div>
                </div>
              )}

              <button
                onClick={handleFileUpload}
                disabled={!uploadFile || uploadProgress}
                className="w-full bg-green-600 text-white rounded-lg px-4 py-2 hover:bg-green-700 disabled:opacity-50 flex items-center justify-center space-x-2"
              >
                {uploadProgress ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4" />
                )}
                <span>Upload File</span>
              </button>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900">Configuration</h3>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Embedding Model
                  </label>
                  <select className="w-full text-sm border border-gray-300 rounded px-3 py-2">
                    <option>mxbai-embed-large</option>
                    <option>nomic-embed-text</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Generative Model
                  </label>
                  <select className="w-full text-sm border border-gray-300 rounded px-3 py-2">
                    <option>codellama</option>
                    <option>llama2</option>
                    <option>mistral</option>
                    <option>phi</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Chunk Size
                  </label>
                  <input
                    type="number"
                    defaultValue={300}
                    className="w-full text-sm border border-gray-300 rounded px-3 py-2"
                  />
                </div>
                
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Chunk Overlap
                  </label>
                  <input
                    type="number"
                    defaultValue={100}
                    className="w-full text-sm border border-gray-300 rounded px-3 py-2"
                  />
                </div>
              </div>
              
              <button className="w-full bg-gray-600 text-white rounded-lg px-4 py-2 hover:bg-gray-700">
                Save Settings
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}