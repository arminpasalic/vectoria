/**
 * Browser-based File Processing
 * Supports CSV, Excel, JSON, and TXT files
 * Pure JavaScript parsing without server backend
 */

import Papa from 'https://cdn.jsdelivr.net/npm/papaparse@5.4.1/+esm';
import * as XLSX from 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/+esm';

export class BrowserFileProcessor {
    constructor() {
        this.supportedFormats = ['csv', 'xlsx', 'xls', 'json', 'txt'];
        this.maxFileSize = 100 * 1024 * 1024; // 100MB
    }

    /**
     * Parse uploaded file
     * @param {File} file - File object from input
     * @returns {Promise<Object>} Parsed data with metadata
     */
    async parseFile(file) {
        // Validate file size
        if (file.size > this.maxFileSize) {
            throw new Error(`File too large. Maximum size: ${this._formatFileSize(this.maxFileSize)}`);
        }

        // Get file extension
        const ext = file.name.split('.').pop().toLowerCase();

        if (!this.supportedFormats.includes(ext)) {
            throw new Error(`Unsupported file format: .${ext}. Supported: ${this.supportedFormats.join(', ')}`);
        }

        // Parse based on file type
        let data, columns;
        switch (ext) {
            case 'csv':
                ({ data, columns } = await this.parseCSV(file));
                break;
            case 'xlsx':
            case 'xls':
                ({ data, columns } = await this.parseExcel(file));
                break;
            case 'json':
                ({ data, columns } = await this.parseJSON(file));
                break;
            case 'txt':
                ({ data, columns } = await this.parseTXT(file));
                break;
            default:
                throw new Error(`Unknown file format: ${ext}`);
        }

        return {
            data: data,
            columns: columns,
            fileName: file.name,
            fileType: ext,
            fileSize: file.size,
            rowCount: data.length
        };
    }

    /**
     * Parse CSV file
     */
    async parseCSV(file) {
        return new Promise((resolve, reject) => {
            const parseTimeout = setTimeout(() => {
                reject(new Error('CSV parsing timed out after 120 seconds'));
            }, 120000);

            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    clearTimeout(parseTimeout);
                    if (results.errors.length > 0) {
                        console.warn('CSV parsing warnings:', results.errors);
                    }

                    const data = results.data;
                    const columns = results.meta.fields || [];

                    resolve({ data, columns });
                },
                error: (error) => {
                    clearTimeout(parseTimeout);
                    reject(new Error(`CSV parsing failed: ${error.message}`));
                }
            });
        });
    }

    /**
     * Parse Excel file (.xlsx, .xls)
     */
    async parseExcel(file) {
        try {
            const buffer = await file.arrayBuffer();
            const workbook = XLSX.read(buffer, { type: 'array' });

            // Get first sheet
            const sheetName = workbook.SheetNames[0];
            const sheet = workbook.Sheets[sheetName];

            // Convert to JSON
            const data = XLSX.utils.sheet_to_json(sheet, { defval: '' });
            const columns = data.length > 0 ? Object.keys(data[0]) : [];

            return { data, columns };
        } catch (error) {
            throw new Error(`Excel parsing failed: ${error.message}`);
        }
    }

    /**
     * Parse JSON file
     */
    async parseJSON(file) {
        try {
            const text = await file.text();
            let parsed = JSON.parse(text);

            // Handle different JSON structures
            let data, columns;

            if (Array.isArray(parsed)) {
                // Array of objects
                data = parsed;
                columns = data.length > 0 ? Object.keys(data[0]) : [];
            } else if (typeof parsed === 'object' && parsed !== null) {
                // Single object - wrap in array
                data = [parsed];
                columns = Object.keys(parsed);
            } else {
                throw new Error('JSON must be an array of objects or a single object');
            }

            return { data, columns };
        } catch (error) {
            throw new Error(`JSON parsing failed: ${error.message}`);
        }
    }

    /**
     * Parse plain text file
     */
    async parseTXT(file) {
        try {
            const text = await file.text();

            // Split into lines
            const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);

            // Create data structure with single 'text' column
            const data = lines.map((line, index) => ({
                text: line,
                line_number: index + 1
            }));

            const columns = ['text', 'line_number'];

            return { data, columns };
        } catch (error) {
            throw new Error(`TXT parsing failed: ${error.message}`);
        }
    }

    /**
     * Extract text from data based on selected column
     * @param {Object[]} data - Parsed data
     * @param {string} textColumn - Column name containing text
     * @returns {Object} Documents with text and metadata, plus empty row count
     */
    extractDocuments(data, textColumn) {
        const documents = [];
        let emptyRowCount = 0;
        const excludedColumns = new Set();

        for (let i = 0; i < data.length; i++) {
            const row = data[i];
            const rawText = row[textColumn];

            let normalizedText = '';
            let isEmpty = false;

            if (rawText === null || rawText === undefined) {
                isEmpty = true;
            } else if (typeof rawText === 'string') {
                normalizedText = rawText.trim();
                const lowerValue = normalizedText.toLowerCase();
                if (normalizedText.length === 0 || lowerValue === 'nan') {
                    isEmpty = true;
                }
            } else if (typeof rawText === 'number') {
                if (Number.isNaN(rawText)) {
                    isEmpty = true;
                } else {
                    normalizedText = String(rawText);
                }
            } else {
                normalizedText = String(rawText || '').trim();
                const lowerValue = normalizedText.toLowerCase();
                if (normalizedText.length === 0 || lowerValue === 'nan') {
                    isEmpty = true;
                }
            }

            if (isEmpty) {
                emptyRowCount++;
            }

            // Create document with text and all other columns as metadata
            // Exclude: text column, empty column names, index columns
            const metadata = {};
            for (const [key, value] of Object.entries(row)) {
                // Skip if it's the text column
                if (key === textColumn) continue;

                // Skip empty or whitespace-only column names
                if (!key || key.trim() === '') {
                    if (i === 0) excludedColumns.add('(empty column name)');
                    continue;
                }

                // Skip index-like column names (e.g., "Unnamed: 0", "__index__", or pure numbers)
                const keyLower = key.toLowerCase().trim();
                if (keyLower.startsWith('unnamed:') ||
                    keyLower === '__index__' ||
                    keyLower === 'index' ||
                    /^[0-9]+$/.test(key)) {
                    if (i === 0) excludedColumns.add(key);
                    continue;
                }

                metadata[key] = value;
            }

            documents.push({
                id: `doc_${i}`,
                text: isEmpty ? '' : normalizedText,
                metadata: metadata,
                originalIndex: i,
                hasEmptyText: isEmpty
            });
        }

        if (excludedColumns.size > 0) {
        }

        return { documents, emptyRowCount };
    }

    /**
     * Get sample data for preview
     * @param {Object[]} data - Full dataset
     * @param {number} sampleSize - Number of rows to sample
     */
    getSample(data, sampleSize = 5) {
        return data.slice(0, Math.min(sampleSize, data.length));
    }

    /**
     * Validate text column
     * @param {Object[]} data
     * @param {string} columnName
     */
    validateTextColumn(data, columnName) {
        if (data.length === 0) {
            throw new Error('Dataset is empty');
        }

        if (!(columnName in data[0])) {
            throw new Error(`Column "${columnName}" not found in data`);
        }

        // Check if column has text content
        let textCount = 0;
        for (let i = 0; i < Math.min(100, data.length); i++) {
            const value = data[i][columnName];
            if (value && typeof value === 'string' && value.trim().length > 0) {
                textCount++;
            }
        }

        if (textCount === 0) {
            throw new Error(`Column "${columnName}" appears to be empty or not text`);
        }

        const textRatio = textCount / Math.min(100, data.length);
        if (textRatio < 0.5) {
            console.warn(`Warning: Column "${columnName}" has low text content (${Math.round(textRatio * 100)}%)`);
        }

        return true;
    }

    /**
     * Format file size for display
     */
    _formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    /**
     * Get column statistics
     * @param {Object[]} data
     * @param {string} columnName
     */
    getColumnStats(data, columnName) {
        let totalLength = 0;
        let nonEmpty = 0;
        let minLength = Infinity;
        let maxLength = 0;

        for (const row of data) {
            const value = row[columnName];
            if (value && typeof value === 'string') {
                const length = value.trim().length;
                if (length > 0) {
                    nonEmpty++;
                    totalLength += length;
                    minLength = Math.min(minLength, length);
                    maxLength = Math.max(maxLength, length);
                }
            }
        }

        return {
            totalRows: data.length,
            nonEmptyRows: nonEmpty,
            emptyRows: data.length - nonEmpty,
            avgLength: nonEmpty > 0 ? Math.round(totalLength / nonEmpty) : 0,
            minLength: minLength === Infinity ? 0 : minLength,
            maxLength: maxLength
        };
    }
}

// Export singleton instance
export const fileProcessor = new BrowserFileProcessor();
