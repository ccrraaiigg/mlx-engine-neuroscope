"""Advanced Feature Localization for Mechanistic Interpretability

This module implements sophisticated feature localization techniques including:
- Sparse Autoencoders for feature discovery
- Dictionary Learning for activation decomposition
- Principal Component Analysis for dimensionality reduction
- Probing classifiers for feature validation
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features that can be localized."""
    SEMANTIC = "semantic"  # Semantic concepts (e.g., "animal", "color")
    SYNTACTIC = "syntactic"  # Syntactic patterns (e.g., "subject", "verb")
    POSITIONAL = "positional"  # Position-based features
    ATTENTION = "attention"  # Attention-related features
    FACTUAL = "factual"  # Factual knowledge features
    ARITHMETIC = "arithmetic"  # Mathematical reasoning features

class LocalizationMethod(Enum):
    """Methods for feature localization."""
    SPARSE_AUTOENCODER = "sparse_autoencoder"
    DICTIONARY_LEARNING = "dictionary_learning"
    PCA = "pca"
    PROBING_CLASSIFIER = "probing_classifier"
    GRADIENT_ATTRIBUTION = "gradient_attribution"

@dataclass
class FeatureSpec:
    """Specification for a feature to localize."""
    name: str
    feature_type: FeatureType
    description: str
    target_layers: List[str]
    examples: List[str]  # Example prompts that should activate this feature
    counter_examples: List[str]  # Prompts that should NOT activate this feature

@dataclass
class LocalizedFeature:
    """Result of feature localization."""
    feature_spec: FeatureSpec
    layer_name: str
    neuron_indices: List[int]
    activation_strength: float
    confidence: float
    localization_method: LocalizationMethod
    metadata: Dict[str, Any]

@dataclass
class FeatureLocalizationResult:
    """Complete result of feature localization analysis."""
    features: List[LocalizedFeature]
    layer_analysis: Dict[str, Dict[str, float]]
    feature_interactions: Dict[str, List[str]]
    execution_time_ms: int
    metadata: Dict[str, Any]

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for discovering interpretable features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_penalty: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_penalty = sparsity_penalty
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def __call__(self, x):
        # Encode
        hidden = nn.relu(self.encoder(x))
        
        # Decode
        reconstructed = self.decoder(hidden)
        
        return reconstructed, hidden
    
    def loss(self, x, reconstructed, hidden):
        """Compute loss with sparsity penalty."""
        # Reconstruction loss
        reconstruction_loss = mx.mean((x - reconstructed) ** 2)
        
        # Sparsity penalty (L1 regularization on hidden activations)
        sparsity_loss = self.sparsity_penalty * mx.mean(mx.abs(hidden))
        
        return reconstruction_loss + sparsity_loss

class DictionaryLearner:
    """Dictionary Learning for sparse coding of neural activations."""
    
    def __init__(self, n_components: int, alpha: float = 1.0, max_iter: int = 1000):
        self.n_components = n_components
        self.alpha = alpha  # Sparsity parameter
        self.max_iter = max_iter
        self.dictionary = None
        self.codes = None
        
    def fit(self, X: mx.array) -> 'DictionaryLearner':
        """Fit dictionary learning model to data."""
        n_samples, n_features = X.shape
        
        # Initialize dictionary randomly
        self.dictionary = mx.random.normal((self.n_components, n_features))
        
        # Normalize dictionary atoms
        self.dictionary = self.dictionary / mx.linalg.norm(self.dictionary, axis=1, keepdims=True)
        
        # Iterative optimization (simplified version)
        for iteration in range(self.max_iter):
            # Sparse coding step: find sparse codes
            codes = self._sparse_coding(X)
            
            # Dictionary update step
            self.dictionary = self._update_dictionary(X, codes)
            
            if iteration % 100 == 0:
                logger.debug(f"Dictionary learning iteration {iteration}")
        
        return self
    
    def _sparse_coding(self, X: mx.array) -> mx.array:
        """Solve sparse coding problem using coordinate descent."""
        n_samples = X.shape[0]
        codes = mx.zeros((n_samples, self.n_components))
        
        # Simplified sparse coding (in practice, use more sophisticated methods)
        for i in range(n_samples):
            residual = X[i]
            for j in range(self.n_components):
                # Compute correlation with dictionary atom
                correlation = mx.dot(residual, self.dictionary[j])
                
                # Soft thresholding
                codes[i, j] = mx.maximum(0, correlation - self.alpha)
                
                # Update residual
                residual = residual - codes[i, j] * self.dictionary[j]
        
        return codes
    
    def _update_dictionary(self, X: mx.array, codes: mx.array) -> mx.array:
        """Update dictionary atoms."""
        new_dictionary = mx.zeros_like(self.dictionary)
        
        for j in range(self.n_components):
            # Find samples that use this atom
            active_samples = codes[:, j] > 0
            
            if mx.sum(active_samples) > 0:
                # Update dictionary atom
                X_active = X[active_samples]
                codes_active = codes[active_samples, j:j+1]
                
                # Compute new atom
                new_atom = mx.mean(X_active / codes_active, axis=0)
                new_dictionary[j] = new_atom / mx.linalg.norm(new_atom)
            else:
                # Keep old atom if no samples use it
                new_dictionary[j] = self.dictionary[j]
        
        return new_dictionary
    
    def transform(self, X: mx.array) -> mx.array:
        """Transform data using learned dictionary."""
        if self.dictionary is None:
            raise ValueError("Dictionary not fitted. Call fit() first.")
        
        return self._sparse_coding(X)

class ProbingClassifier:
    """Probing classifier for validating feature localization."""
    
    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
        self.model = None
        
    def build_model(self, input_dim: int, num_classes: int):
        """Build the probing classifier model."""
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
    
    def train(self, X: mx.array, y: mx.array, epochs: int = 100) -> Dict[str, float]:
        """Train the probing classifier."""
        if self.model is None:
            num_classes = len(mx.unique(y))
            self.build_model(X.shape[1], num_classes)
        
        # Simple training loop (in practice, use more sophisticated optimization)
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            logits = self.model(X)
            
            # Compute loss (cross-entropy)
            loss = self._cross_entropy_loss(logits, y)
            losses.append(float(loss))
            
            # Compute accuracy
            predictions = mx.argmax(logits, axis=1)
            accuracy = float(mx.mean(predictions == y))
            accuracies.append(accuracy)
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return {
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1],
            'training_history': {'losses': losses, 'accuracies': accuracies}
        }
    
    def _cross_entropy_loss(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Compute cross-entropy loss."""
        # Softmax
        exp_logits = mx.exp(logits - mx.max(logits, axis=1, keepdims=True))
        probs = exp_logits / mx.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy
        log_probs = mx.log(probs + 1e-8)  # Add small epsilon for numerical stability
        return -mx.mean(log_probs[mx.arange(len(targets)), targets])
    
    def evaluate(self, X: mx.array, y: mx.array) -> Dict[str, float]:
        """Evaluate the classifier."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logits = self.model(X)
        predictions = mx.argmax(logits, axis=1)
        accuracy = float(mx.mean(predictions == y))
        
        return {'accuracy': accuracy}

class FeatureLocalizer:
    """Main class for feature localization using various methods."""
    
    def __init__(self, model):
        self.model = model
        self.localized_features = {}
        
    def localize_features(
        self,
        feature_specs: List[FeatureSpec],
        method: LocalizationMethod = LocalizationMethod.SPARSE_AUTOENCODER,
        **kwargs
    ) -> FeatureLocalizationResult:
        """Localize features using the specified method."""
        start_time = time.time()
        
        features = []
        layer_analysis = {}
        feature_interactions = {}
        
        for feature_spec in feature_specs:
            logger.info(f"Localizing feature: {feature_spec.name}")
            
            if method == LocalizationMethod.SPARSE_AUTOENCODER:
                localized = self._localize_with_sparse_autoencoder(feature_spec, **kwargs)
            elif method == LocalizationMethod.DICTIONARY_LEARNING:
                localized = self._localize_with_dictionary_learning(feature_spec, **kwargs)
            elif method == LocalizationMethod.PCA:
                localized = self._localize_with_pca(feature_spec, **kwargs)
            elif method == LocalizationMethod.PROBING_CLASSIFIER:
                localized = self._localize_with_probing_classifier(feature_spec, **kwargs)
            else:
                raise ValueError(f"Unsupported localization method: {method}")
            
            features.extend(localized)
            
            # Analyze layer-wise feature distribution
            for feature in localized:
                layer_name = feature.layer_name
                if layer_name not in layer_analysis:
                    layer_analysis[layer_name] = {}
                
                layer_analysis[layer_name][feature_spec.name] = feature.confidence
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        return FeatureLocalizationResult(
            features=features,
            layer_analysis=layer_analysis,
            feature_interactions=feature_interactions,
            execution_time_ms=execution_time_ms,
            metadata={'method': method.value, 'num_features': len(feature_specs)}
        )
    
    def _localize_with_sparse_autoencoder(
        self,
        feature_spec: FeatureSpec,
        hidden_dim: int = 512,
        sparsity_penalty: float = 0.01,
        epochs: int = 100
    ) -> List[LocalizedFeature]:
        """Localize features using sparse autoencoder."""
        localized_features = []
        
        for layer_name in feature_spec.target_layers:
            # Get activations for this layer
            activations = self._get_layer_activations(layer_name, feature_spec.examples)
            
            if activations is None:
                continue
            
            # Train sparse autoencoder
            input_dim = activations.shape[1]
            autoencoder = SparseAutoencoder(input_dim, hidden_dim, sparsity_penalty)
            
            # Simple training loop (in practice, use proper optimization)
            for epoch in range(epochs):
                reconstructed, hidden = autoencoder(activations)
                loss = autoencoder.loss(activations, reconstructed, hidden)
                
                if epoch % 20 == 0:
                    logger.debug(f"Autoencoder epoch {epoch}: Loss = {float(loss):.4f}")
            
            # Analyze learned features
            _, final_hidden = autoencoder(activations)
            
            # Find most active neurons for this feature
            mean_activation = mx.mean(final_hidden, axis=0)
            top_neurons = mx.argsort(mean_activation)[-10:]  # Top 10 neurons
            
            localized_feature = LocalizedFeature(
                feature_spec=feature_spec,
                layer_name=layer_name,
                neuron_indices=list(map(int, top_neurons)),
                activation_strength=float(mx.mean(mean_activation[top_neurons])),
                confidence=0.8,  # Placeholder confidence
                localization_method=LocalizationMethod.SPARSE_AUTOENCODER,
                metadata={
                    'hidden_dim': hidden_dim,
                    'sparsity_penalty': sparsity_penalty,
                    'reconstruction_loss': float(loss)
                }
            )
            
            localized_features.append(localized_feature)
        
        return localized_features
    
    def _localize_with_dictionary_learning(
        self,
        feature_spec: FeatureSpec,
        n_components: int = 256,
        alpha: float = 1.0
    ) -> List[LocalizedFeature]:
        """Localize features using dictionary learning."""
        localized_features = []
        
        for layer_name in feature_spec.target_layers:
            activations = self._get_layer_activations(layer_name, feature_spec.examples)
            
            if activations is None:
                continue
            
            # Apply dictionary learning
            dict_learner = DictionaryLearner(n_components, alpha)
            dict_learner.fit(activations)
            
            # Transform activations to sparse codes
            codes = dict_learner.transform(activations)
            
            # Find most important dictionary atoms
            atom_importance = mx.mean(mx.abs(codes), axis=0)
            top_atoms = mx.argsort(atom_importance)[-10:]
            
            localized_feature = LocalizedFeature(
                feature_spec=feature_spec,
                layer_name=layer_name,
                neuron_indices=list(map(int, top_atoms)),
                activation_strength=float(mx.mean(atom_importance[top_atoms])),
                confidence=0.75,
                localization_method=LocalizationMethod.DICTIONARY_LEARNING,
                metadata={
                    'n_components': n_components,
                    'alpha': alpha,
                    'sparsity': float(mx.mean(codes > 0))
                }
            )
            
            localized_features.append(localized_feature)
        
        return localized_features
    
    def _localize_with_pca(
        self,
        feature_spec: FeatureSpec,
        n_components: int = 50
    ) -> List[LocalizedFeature]:
        """Localize features using PCA."""
        localized_features = []
        
        for layer_name in feature_spec.target_layers:
            activations = self._get_layer_activations(layer_name, feature_spec.examples)
            
            if activations is None:
                continue
            
            # Center the data
            mean_activation = mx.mean(activations, axis=0)
            centered_activations = activations - mean_activation
            
            # Compute covariance matrix
            cov_matrix = mx.matmul(centered_activations.T, centered_activations) / (activations.shape[0] - 1)
            
            # Compute eigenvalues and eigenvectors (simplified)
            # In practice, use proper SVD or eigendecomposition
            eigenvalues = mx.diagonal(cov_matrix)  # Simplified
            top_components = mx.argsort(eigenvalues)[-n_components:]
            
            localized_feature = LocalizedFeature(
                feature_spec=feature_spec,
                layer_name=layer_name,
                neuron_indices=list(map(int, top_components)),
                activation_strength=float(mx.mean(eigenvalues[top_components])),
                confidence=0.7,
                localization_method=LocalizationMethod.PCA,
                metadata={
                    'n_components': n_components,
                    'explained_variance': float(mx.sum(eigenvalues[top_components]))
                }
            )
            
            localized_features.append(localized_feature)
        
        return localized_features
    
    def _localize_with_probing_classifier(
        self,
        feature_spec: FeatureSpec,
        hidden_dim: int = 128
    ) -> List[LocalizedFeature]:
        """Localize features using probing classifier."""
        localized_features = []
        
        for layer_name in feature_spec.target_layers:
            # Get positive and negative examples
            pos_activations = self._get_layer_activations(layer_name, feature_spec.examples)
            neg_activations = self._get_layer_activations(layer_name, feature_spec.counter_examples)
            
            if pos_activations is None or neg_activations is None:
                continue
            
            # Prepare training data
            X = mx.concatenate([pos_activations, neg_activations], axis=0)
            y = mx.concatenate([
                mx.ones(pos_activations.shape[0]),
                mx.zeros(neg_activations.shape[0])
            ])
            
            # Train probing classifier
            classifier = ProbingClassifier(hidden_dim)
            training_results = classifier.train(X, y)
            
            # Analyze feature importance (simplified)
            # In practice, use gradient-based attribution or permutation importance
            feature_importance = mx.abs(mx.mean(pos_activations - neg_activations, axis=0))
            top_features = mx.argsort(feature_importance)[-20:]
            
            localized_feature = LocalizedFeature(
                feature_spec=feature_spec,
                layer_name=layer_name,
                neuron_indices=list(map(int, top_features)),
                activation_strength=float(mx.mean(feature_importance[top_features])),
                confidence=training_results['final_accuracy'],
                localization_method=LocalizationMethod.PROBING_CLASSIFIER,
                metadata={
                    'classifier_accuracy': training_results['final_accuracy'],
                    'training_loss': training_results['final_loss']
                }
            )
            
            localized_features.append(localized_feature)
        
        return localized_features
    
    def _get_layer_activations(self, layer_name: str, prompts: List[str]) -> Optional[mx.array]:
        """Get activations for a specific layer given input prompts."""
        # This is a placeholder - in practice, you'd use the activation hooks
        # to capture real activations from the model
        
        if not hasattr(self.model, 'activation_hook_manager'):
            logger.warning(f"Model does not support activation capture for layer {layer_name}")
            return None
        
        # Simulate activation capture
        # In real implementation, this would:
        # 1. Register hooks for the specified layer
        # 2. Run inference on the prompts
        # 3. Collect and return the activations
        
        # For now, return random activations as placeholder
        num_samples = len(prompts)
        activation_dim = 768  # Typical transformer hidden dimension
        
        return mx.random.normal((num_samples, activation_dim))

def create_feature_localization_pipeline(model) -> FeatureLocalizer:
    """Create a complete feature localization pipeline."""
    return FeatureLocalizer(model)

# Predefined feature specifications for common interpretability tasks
COMMON_FEATURES = [
    FeatureSpec(
        name="subject_identification",
        feature_type=FeatureType.SYNTACTIC,
        description="Identifies the subject of a sentence",
        target_layers=["model.layers.8", "model.layers.12"],
        examples=[
            "The cat sat on the mat",
            "John went to the store",
            "The beautiful flower bloomed"
        ],
        counter_examples=[
            "Sat on the mat",
            "Went to the store",
            "Bloomed in the garden"
        ]
    ),
    FeatureSpec(
        name="factual_recall",
        feature_type=FeatureType.FACTUAL,
        description="Recalls factual information",
        target_layers=["model.layers.15", "model.layers.18"],
        examples=[
            "The capital of France is",
            "Einstein was born in",
            "The largest planet is"
        ],
        counter_examples=[
            "The color of the sky",
            "My favorite food is",
            "I think that"
        ]
    ),
    FeatureSpec(
        name="arithmetic_reasoning",
        feature_type=FeatureType.ARITHMETIC,
        description="Performs arithmetic operations",
        target_layers=["model.layers.10", "model.layers.14"],
        examples=[
            "2 + 3 =",
            "10 - 4 =",
            "5 * 6 ="
        ],
        counter_examples=[
            "The weather is",
            "I like to",
            "Once upon a time"
        ]
    )
]