export const API_BASE = 'http://localhost:5000/api';

export const api = {
    async analyzeGenome(file: File, disease: string = '') {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('disease', disease);

        try {
            const response = await fetch(`${API_BASE}/analyze-genome`, {
                method: 'POST',
                body: formData
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    async analyzeCompatibility(data: {
        drug_name: string;
        disease: string;
        patient_data: any;
    }) {
        try {
            const response = await fetch(`${API_BASE}/analyze-compatibility`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    async submitReport(data: any) {
        try {
            const response = await fetch(`${API_BASE}/submit-report`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }
};
