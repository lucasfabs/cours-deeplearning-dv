const canvas = document.getElementById('chiffreCanvas');
const ctx = canvas.getContext('2d');
const clearButton = document.querySelector('.clearButton');
const modelStatus = document.getElementById('modelStatus');
const predictButton = document.getElementById('predictButton');
const predictionResult = document.getElementById('predictionResult');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Configuration du canvas : fond blanc, trait noir
ctx.fillStyle = '#ffffff';
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = '#000'; // Couleur du trait : Noir
ctx.lineWidth = 12;          // Épaisseur du trait
ctx.lineCap = 'round';       // Extrémités des lignes arrondies
ctx.lineJoin = 'round';      // Jointures des lignes arrondies

function draw(e) {
    if (!isDrawing) return; // Arrête la fonction si nous ne sommes pas en train de dessiner

    // Calcule la position de la souris par rapport au canvas
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.beginPath(); // Commence un nouveau chemin
    // Déplace le point de départ vers les dernières coordonnées enregistrées
    ctx.moveTo(lastX, lastY);
    // Dessine une ligne jusqu'aux coordonnées actuelles
    ctx.lineTo(x, y);
    ctx.stroke(); // Applique le tracé

    // Met à jour les dernières coordonnées pour le prochain segment
    [lastX, lastY] = [x, y];
}

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    // Enregistre le point de départ du dessin
    [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];
    draw(e); // Dessine un point unique au clic
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    // Efface tout le contenu du canvas, du point (0,0) à sa largeur et hauteur maximales
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Réinitialise le fond blanc
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#000';
    
    // Effacer aussi le résultat de prédiction
    if (predictionResult) {
        predictionResult.innerHTML = '';
        predictionResult.className = 'prediction-result';
    }
}

// ----------------------- model --------------------------------

let session = null; // Variable pour stocker la session ONNX

// Fonction pour charger le modèle ONNX automatiquement
async function loadModel() {
    try {
        // Afficher le statut de chargement
        if (modelStatus) {
            modelStatus.textContent = 'Chargement du modèle...';
            modelStatus.className = 'model-status';
        }

        // Charger le modèle depuis le dossier
        const response = await fetch('model.onnx');
        if (!response.ok) {
            throw new Error('Impossible de charger le modèle. Assurez-vous d\'utiliser un serveur web local (ex: npm start) et que le fichier "model.onnx" existe dans le dossier.');
        }
        
        const arrayBuffer = await response.arrayBuffer();
        
        // Créer une session ONNX Runtime
        session = await ort.InferenceSession.create(arrayBuffer);
        
        // Afficher le succès
        if (modelStatus) {
            modelStatus.textContent = 'Modèle chargé avec succès ✓';
            modelStatus.className = 'model-status success';
        }
        
        console.log('Modèle chargé:', session);
    } catch (error) {
        console.error('Erreur lors du chargement du modèle:', error);
        if (modelStatus) {
            let errorMessage = error.message;
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                errorMessage = 'Erreur: Vous devez utiliser un serveur web local. Lancez "npm start" dans le terminal.';
            }
            modelStatus.textContent = errorMessage;
            modelStatus.className = 'model-status error';
        }
        session = null;
    }
}

// Fonction pour convertir le canvas en tensor 28x28
function canvasToTensor() {
    // Créer un canvas temporaire de 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Dessiner le canvas original redimensionné sur le canvas temporaire
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Récupérer les données de l'image
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Convertir en tensor (format attendu: [1, 1, 28, 28])
    // Pour MNIST, on utilise uniquement le canal rouge (ou la moyenne des canaux)
    // et on normalise entre 0 et 1, puis on inverse (fond blanc = 0, trait noir = 1)
    const tensor = new Float32Array(1 * 1 * 28 * 28);
    
    for (let i = 0; i < 28 * 28; i++) {
        const r = data[i * 4];     // Rouge
        const g = data[i * 4 + 1]; // Vert
        const b = data[i * 4 + 2]; // Bleu
        // Convertir en niveaux de gris et normaliser (0-255 -> 0-1)
        // Inverser: blanc (255) -> 0, noir (0) -> 1
        const gray = (r + g + b) / 3;
        tensor[i] = (255 - gray) / 255.0;
    }
    
    return new ort.Tensor('float32', tensor, [1, 1, 28, 28]);
}

// Fonction pour faire la prédiction
async function predict() {
    // Vérifier que le modèle est chargé
    if (!session) {
        predictionResult.innerHTML = '<div class="prediction-error">Veuillez d\'abord charger un modèle</div>';
        predictionResult.className = 'prediction-result error';
        return;
    }
    
    // Vérifier que le canvas n'est pas vide
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    let hasDrawing = false;
    
    for (let i = 0; i < data.length; i += 4) {
        // Si un pixel n'est pas blanc, il y a un dessin
        if (data[i] !== 255 || data[i + 1] !== 255 || data[i + 2] !== 255) {
            hasDrawing = true;
            break;
        }
    }
    
    if (!hasDrawing) {
        predictionResult.innerHTML = '<div class="prediction-error">Veuillez dessiner un chiffre</div>';
        predictionResult.className = 'prediction-result error';
        return;
    }
    
    try {
        // Afficher le chargement
        predictionResult.innerHTML = '<div class="prediction-label">Prédiction en cours...</div>';
        predictionResult.className = 'prediction-result';
        
        // Convertir le canvas en tensor
        const inputTensor = canvasToTensor();
        
        // Récupérer le nom de l'input du modèle
        const inputName = session.inputNames[0];
        
        // Faire l'inférence
        const feeds = { [inputName]: inputTensor };
        const results = await session.run(feeds);
        
        // Récupérer les résultats (logits ou probabilités pour chaque chiffre 0-9)
        const output = results[session.outputNames[0]];
        const logits = Array.from(output.data);
        
        // Appliquer softmax pour convertir les logits en probabilités
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit)); // Soustraction pour stabilité numérique
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probabilities = expLogits.map(x => x / sumExp);
        
        // Trouver le chiffre avec la probabilité la plus élevée
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const confidence = probabilities[maxIndex];
        
        // Afficher le résultat
        const confidencePercent = (confidence * 100).toFixed(1);
        
        predictionResult.innerHTML = `
            <div class="prediction-content">
                <div class="prediction-label">Chiffre prédit</div>
                <div class="prediction-digit">${maxIndex}</div>
                <div class="prediction-confidence">Confiance: ${confidencePercent}%</div>
            </div>
        `;
        predictionResult.className = 'prediction-result success';
        
    } catch (error) {
        console.error('Erreur lors de la prédiction:', error);
        predictionResult.innerHTML = '<div class="prediction-error">Erreur: ' + error.message + '</div>';
        predictionResult.className = 'prediction-result error';
    }
}



// --- Événements ---

// Dessin : Utilisation de 'pointerevents' pour une meilleure compatibilité (souris/tactile)
canvas.addEventListener('pointerdown', startDrawing);
canvas.addEventListener('pointermove', draw);
canvas.addEventListener('pointerup', stopDrawing);
canvas.addEventListener('pointerout', stopDrawing); // Arrête le dessin si la souris sort du canvas

// Effacement : Au clic sur le bouton
if (clearButton) {
    clearButton.addEventListener('click', clearCanvas);
}

// Chargement automatique du modèle au chargement de la page
loadModel();

// Prédiction : Au clic sur le bouton Prédire
if (predictButton) {
    predictButton.addEventListener('click', predict);
}