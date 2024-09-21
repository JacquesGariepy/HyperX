import argparse
import json
import logging
import re
from typing import List, Dict, Any
import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import geoopt
from datasets import load_dataset
from transformers import AutoTokenizer 
from tqdm import tqdm

from datasets import concatenate_datasets

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérification de la disponibilité de CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

class HyperbolicEmbedding(nn.Module):
    """
    Classe pour les embeddings hyperboliques utilisant la géométrie de Poincaré.

    Args:
        vocab_size (int): Taille du vocabulaire, c'est-à-dire le nombre total de tokens.
        embed_dim (int): Dimension des embeddings, c'est-à-dire la taille de chaque vecteur d'embedding.
        manifold_radius (float, optionnel): Rayon de la sphère de Poincaré. Par défaut à 1.0.
        init_method (str, optionnel): Méthode d'initialisation des embeddings. Peut être "xavier_uniform", "xavier_normal" ou "random". Par défaut à "xavier_uniform".
    """
    def __init__(self, vocab_size: int, embed_dim: int, manifold_radius: float = 1.0, init_method: str = "xavier_uniform"):
        super().__init__()
        
        # Initialiser la sphère de Poincaré avec une courbure c = 1 / (manifold_radius^2)
        self.manifold = geoopt.PoincareBall(c=1.0 / (manifold_radius ** 2))
        
        # Initialiser le tenseur d'embeddings selon la méthode spécifiée
        if init_method == "xavier_uniform":
            # Initialisation Xavier uniforme
            init_tensor = torch.empty(vocab_size, embed_dim)
            nn.init.xavier_uniform_(init_tensor)
        elif init_method == "xavier_normal":
            # Initialisation Xavier normale
            init_tensor = torch.empty(vocab_size, embed_dim)
            nn.init.xavier_normal_(init_tensor)
        else:
            # Initialisation aléatoire normale
            init_tensor = torch.randn(vocab_size, embed_dim)
        
        # Créer un paramètre de manifold pour les embeddings
        self.embedding = geoopt.ManifoldParameter(init_tensor, manifold=self.manifold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant pour obtenir les embeddings des indices de tokens.

        Args:
            x (torch.Tensor): Tensor contenant les indices des tokens.

        Returns:
            torch.Tensor: Tensor contenant les embeddings correspondants.
        """
        return self.embedding[x]

class ScaledDotProductAttention(nn.Module):
    """
    Classe pour l'attention par produit scalaire avec mise à l'échelle.

    Args:
        temperature (float): Facteur de mise à l'échelle pour les scores d'attention.
    """
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature  # Stocke le facteur de mise à l'échelle

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagation avant pour calculer les scores d'attention et les pondérations.

        Args:
            q (torch.Tensor): Tensor des requêtes (queries) de dimension (batch_size, num_heads, seq_len_q, d_k).
            k (torch.Tensor): Tensor des clés (keys) de dimension (batch_size, num_heads, seq_len_k, d_k).
            v (torch.Tensor): Tensor des valeurs (values) de dimension (batch_size, num_heads, seq_len_v, d_v).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple contenant le tensor de sortie et les pondérations d'attention.
        """
        # Calculer les scores d'attention en multipliant les requêtes par les clés transposées, puis en les divisant par la température
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        
        # Appliquer la fonction softmax pour obtenir les pondérations d'attention
        attn = F.softmax(attn, dim=-1)
        
        # Calculer la sortie en multipliant les pondérations d'attention par les valeurs
        output = torch.matmul(attn, v)
        
        return output, attn  # Retourner la sortie et les pondérations d'attention

class MultiHeadAttention(nn.Module):
    """
    Classe pour l'attention multi-tête.

    Args:
        embed_dim (int): Dimension des embeddings, c'est-à-dire la taille de chaque vecteur d'embedding.
        num_heads (int): Nombre de têtes d'attention.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads  # Stocke le nombre de têtes d'attention
        self.embed_dim = embed_dim  # Stocke la dimension des embeddings
        self.head_dim = embed_dim // num_heads  # Calcule la dimension de chaque tête
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"  # Vérifie que embed_dim est divisible par num_heads
        
        # Crée des couches linéaires pour les requêtes, clés et valeurs
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Initialise l'attention par produit scalaire avec mise à l'échelle
        self.attention = ScaledDotProductAttention(temperature=self.head_dim ** 0.5)
        
        # Crée une couche linéaire pour la sortie
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagation avant pour calculer l'attention multi-tête.

        Args:
            q (torch.Tensor): Tensor des requêtes (queries) de dimension (batch_size, seq_len, embed_dim).
            k (torch.Tensor): Tensor des clés (keys) de dimension (batch_size, seq_len, embed_dim).
            v (torch.Tensor): Tensor des valeurs (values) de dimension (batch_size, seq_len, embed_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple contenant le tensor de sortie et les pondérations d'attention.
        """
        batch_size = q.size(0)  # Obtient la taille du batch
        
        # Applique les couches linéaires et reformate les tensors pour les requêtes, clés et valeurs
        # Applique une transformation linéaire aux requêtes (queries) et reformate le tensor
        # 1. self.q_linear(q): Applique une couche linéaire pour projeter les requêtes dans un espace de dimension embed_dim.
        # 2. view(batch_size, -1, self.num_heads, self.head_dim): Change la forme du tensor pour qu'il ait les dimensions (batch_size, seq_len, num_heads, head_dim).
        # 3. transpose(1, 2): Transpose les dimensions pour obtenir la forme (batch_size, num_heads, seq_len, head_dim), ce qui permet de séparer les têtes d'attention.
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Applique une transformation linéaire aux clés (keys) et reformate le tensor
        # 1. self.k_linear(k): Applique une couche linéaire pour projeter les clés dans un espace de dimension embed_dim.
        # 2. view(batch_size, -1, self.num_heads, self.head_dim): Change la forme du tensor pour qu'il ait les dimensions (batch_size, seq_len, num_heads, head_dim).
        # 3. transpose(1, 2): Transpose les dimensions pour obtenir la forme (batch_size, num_heads, seq_len, head_dim), ce qui permet de séparer les têtes d'attention.
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Applique une transformation linéaire aux valeurs (values) et reformate le tensor
        # 1. self.v_linear(v): Applique une couche linéaire pour projeter les valeurs dans un espace de dimension embed_dim.
        # 2. view(batch_size, -1, self.num_heads, self.head_dim): Change la forme du tensor pour qu'il ait les dimensions (batch_size, seq_len, num_heads, head_dim).
        # 3. transpose(1, 2): Transpose les dimensions pour obtenir la forme (batch_size, num_heads, seq_len, head_dim), ce qui permet de séparer les têtes d'attention.
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcule l'attention par produit scalaire avec mise à l'échelle
        attn_output, attn_weights = self.attention(q, k, v)
        
        # Reformate le tensor de sortie et applique la couche linéaire de sortie
        # Reformate le tensor de sortie de l'attention et applique une transformation linéaire
        # 1. transpose(1, 2): Transpose les dimensions du tensor pour obtenir la forme (batch_size, seq_len, num_heads * head_dim).
        # 2. contiguous(): Assure que le tensor est contigu en mémoire après la transposition.
        # 3. view(batch_size, -1, self.embed_dim): Change la forme du tensor pour qu'il ait les dimensions (batch_size, seq_len, embed_dim).
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(attn_output)
        
        return output, attn_weights  # Retourne la sortie et les pondérations d'attention

class HyperbolicAttentionLayer(nn.Module):
    """
    Classe pour une couche d'attention hyperbolique.

    Args:
        embed_dim (int): Dimension des embeddings, c'est-à-dire la taille de chaque vecteur d'embedding.
        dropout (float, optionnel): Taux de dropout à appliquer. Par défaut à 0.1.
        layer_norm (bool, optionnel): Indique si la normalisation de couche doit être appliquée. Par défaut à True.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, layer_norm: bool = True):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads=8)  # Initialise l'attention multi-tête
        self.feed_forward = nn.Sequential(  # Initialise le réseau feed-forward
            nn.Linear(embed_dim, 4 * embed_dim),  # Couche linéaire avec une expansion de 4 fois embed_dim
            nn.ReLU(),  # Fonction d'activation ReLU
            nn.Linear(4 * embed_dim, embed_dim)  # Couche linéaire pour réduire la dimension à embed_dim
        )
        self.dropout1 = nn.Dropout(dropout)  # Dropout après l'attention
        self.dropout2 = nn.Dropout(dropout)  # Dropout après le réseau feed-forward
        self.layer_norm1 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()  # Normalisation de couche ou identité
        self.layer_norm2 = nn.LayerNorm(embed_dim) if layer_norm else nn.Identity()  # Normalisation de couche ou identité

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant pour calculer la sortie de la couche d'attention hyperbolique.

        Args:
            x (torch.Tensor): Tensor d'entrée de dimension (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Tensor de sortie de dimension (batch_size, seq_len, embed_dim).
        """
        attn_output, _ = self.attention(x, x, x)  # Calcule l'attention multi-tête
        x = self.layer_norm1(x + self.dropout1(attn_output))  # Applique la normalisation de couche et le dropout après l'attention
        ff_output = self.feed_forward(x)  # Passe par le réseau feed-forward
        x = self.layer_norm2(x + self.dropout2(ff_output))  # Applique la normalisation de couche et le dropout après le feed-forward
        return x  # Retourne le tensor de sortie

class HyperbolicAttentionNetwork(nn.Module):
    """
    Classe pour un réseau d'attention hyperbolique.

    Args:
        vocab_size (int): Taille du vocabulaire, c'est-à-dire le nombre total de tokens.
        embed_dim (int): Dimension des embeddings, c'est-à-dire la taille de chaque vecteur d'embedding.
        num_layers (int): Nombre de couches d'attention hyperbolique.
        dropout (float, optionnel): Taux de dropout à appliquer après chaque couche d'attention. Par défaut à 0.1.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        # Initialiser les embeddings hyperboliques
        self.embedding = HyperbolicEmbedding(vocab_size, embed_dim)
        
        # Créer une liste de couches d'attention hyperbolique
        self.layers = nn.ModuleList([
            HyperbolicAttentionLayer(embed_dim, dropout=dropout) for _ in range(num_layers)
        ])
        
        # Créer une couche linéaire pour la sortie
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant pour le réseau d'attention hyperbolique.

        Args:
            x (torch.Tensor): Tensor contenant les indices des tokens.

        Returns:
            torch.Tensor: Tensor contenant les logits de sortie pour chaque token.
        """
        # Appliquer les embeddings hyperboliques
        x = self.embedding(x)
        
        # Appliquer chaque couche d'attention hyperbolique
        for layer in self.layers:
            x = layer(x)
        
        # Appliquer la couche linéaire de sortie
        return self.output_layer(x)

class DialogDataset(Dataset):
    """
    Classe pour le dataset de dialogues.

    Args:
        data (List[Dict[str, Any]]): Liste de dictionnaires contenant les dialogues.
        tokenizer (AutoTokenizer): Tokenizer pour transformer le texte en tokens.
        max_length (int, optionnel): Longueur maximale des séquences de tokens. Par défaut à 32.
    """
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 32):
        self.data = data  # Stocke les données de dialogues
        self.tokenizer = tokenizer  # Stocke le tokenizer
        self.max_length = max_length  # Stocke la longueur maximale des séquences de tokens

    def __len__(self) -> int:
        """
        Retourne la taille du dataset.

        Returns:
            int: Nombre de dialogues dans le dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un élément du dataset à l'index spécifié.

        Args:
            idx (int): Index de l'élément à retourner.

        Returns:
            Dict[str, torch.Tensor]: Dictionnaire contenant les tensors des séquences d'entrée et de cible.
        """
        dialog = self.data[idx]['dialog']  # Récupère le dialogue à l'index spécifié
        if len(dialog) < 2:  # Vérifie que le dialogue contient au moins deux phrases
            return self.__getitem__((idx + 1) % len(self.data))  # Si non, retourne le prochain dialogue

        input_text = dialog[:-1]  # Texte d'entrée (toutes les phrases sauf la dernière)
        target_text = dialog[1:]  # Texte cible (toutes les phrases sauf la première)

        # Tokenize le texte d'entrée
        input_ids = self.tokenizer(
            " ".join(input_text), truncation=True, max_length=self.max_length, padding='max_length', return_tensors="pt"
        )['input_ids'].squeeze()

        # Tokenize le texte cible
        target_ids = self.tokenizer(
            " ".join(target_text), truncation=True, max_length=self.max_length, padding='max_length', return_tensors="pt"
        )['input_ids'].squeeze()

        return {'input': input_ids, 'target': target_ids}  # Retourne les tensors des séquences d'entrée et de cible

def load_datasets(datasets_list: List[str]) -> List[Any]:
    """
    Charge plusieurs datasets à partir de leurs noms.

    Args:
        datasets_list (List[str]): Liste des noms des datasets à charger.

    Returns:
        List[Any]: Liste des datasets chargés.
    """
    all_datasets = []
    for dataset_name in datasets_list:
        # Charge le dataset en utilisant le nom fourni
        dataset = load_dataset(dataset_name, trust_remote_code=True) # Attention à la sécurité ici, assurez-vous de faire confiance au code distant
        # Ajoute le dataset de train à la liste
        if "train" in dataset:
            all_datasets.append(dataset["train"])
        else:
            all_datasets.append(dataset)
    return all_datasets

def create_dataloader_from_multiple_datasets(datasets_list: List[str], tokenizer: AutoTokenizer, batch_size: int) -> DataLoader:
    all_datasets = load_datasets(datasets_list)
    
    # Log information about each dataset
    for i, dataset in enumerate(all_datasets):
        logger.info(f"Dataset {i} structure: {dataset}")
        logger.info(f"Dataset {i} first few examples: {dataset[:5]}")
    
    combined_dataset = concatenate_datasets(all_datasets)
    
    # Filter out None dialogues
    def filter_none_dialogs(example):
        text_column = "dialog" if "dialog" in example else "text"
        return example[text_column] is not None and len(example[text_column]) > 0

    filtered_dataset = combined_dataset.filter(filter_none_dialogs)
    
    logger.info(f"Original dataset size: {len(combined_dataset)}")
    logger.info(f"Filtered dataset size: {len(filtered_dataset)}")
    
    def tokenize_function(examples):
        text_column = "dialog" if "dialog" in examples else "text"

        processed_examples = []
        for dialog in examples[text_column]:
            if isinstance(dialog, list):
                processed_dialog = ' '.join(str(d) for d in dialog if d is not None)
            elif isinstance(dialog, str):
                processed_dialog = dialog
            else:
                logger.warning(f"Unexpected data type in dialog: {type(dialog)}")
                processed_dialog = str(dialog)

            if processed_dialog.strip():  # Only add non-empty dialogs
                processed_examples.append(processed_dialog)

        logger.info(f"Number of processed examples: {len(processed_examples)}")
        logger.info(f"Sample processed examples: {processed_examples[:2] if processed_examples else 'No examples'}")

        if not processed_examples:
            logger.warning("No valid examples found in this batch.")
            return {"input_ids": [], "attention_mask": []}

        # Tokenize with padding and truncation (this happens BEFORE sending to the model)
        tokenized_output = tokenizer(processed_examples, padding=True, truncation=True, max_length=32, return_tensors="pt")

        return {
            'input_ids': tokenized_output['input_ids'].squeeze(),
            'attention_mask': tokenized_output['attention_mask'].squeeze()
        }

    # Dans la fonction create_dataloader_from_multiple_datasets :
    tokenized_dataset = filtered_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=filtered_dataset.column_names,
        batch_size=1000
    )

    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example['input_ids']) > 0)

    logger.info(f"Final tokenized dataset size: {len(tokenized_dataset)}")

    # Updated format setting to match the columns present in the dataset
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Applique une température aux logits pour ajuster la distribution de probabilité.

    Args:
        logits (torch.Tensor): Tensor des logits.
        temperature (float, optionnel): Température à appliquer. Par défaut à 1.0.

    Returns:
        torch.Tensor: Tensor des logits ajustés.
    """
    # Divise les logits par la température si elle n'est pas égale à 1.0
    return logits / temperature if temperature != 1.0 else logits

def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    """
    Applies a repetition penalty to logits to discourage repetitive tokens.

    Args:
        logits (torch.Tensor): The logits output by the model.
        generated_tokens (list): List of tokens already generated.
        penalty (float): The penalty to apply to repeated tokens.

    Returns:
        torch.Tensor: The adjusted logits with the penalty applied.
    """
    for token in set(generated_tokens):
        logits[0, token] /= penalty  # Penalize the repeated token
    return logits

def chatbot_inference(
    model, tokenizer, input_text, max_length=50, temperature=0.7, top_k=50, top_p=0.9, device='cpu', repetition_penalty=1.2):
    """
    Generates a response from the chatbot based on input text with repetition penalty.

    Args:
        model (nn.Module): The trained language model.
        tokenizer (Tokenizer): The tokenizer to encode/decode text.
        input_text (str): The input text for the chatbot.
        max_length (int): Maximum length of the generated response.
        temperature (float): Controls randomness of generation (lower = more deterministic).
        top_k (int): Number of most probable tokens to consider at each step.
        top_p (float): Cumulative probability for nucleus sampling.
        device (str): Device to run the inference ('cpu' or 'cuda').
        repetition_penalty (float): Penalty factor for repeated tokens.

    Returns:
        str: The generated chatbot response.
    """
    try:
        model.eval()

        # Define and encode special tags from the system prompt
        special_tags = {
            "objectives_definition": "<step as=\"objectives_definition\" />",
            "comprehension": "<step as=\"comprehension\" />",
            "knowledge_cutoff": "<step as=\"knowledge_cutoff\" />",
            "stakeholders_identification": "<step as=\"stakeholders_identification\" />",
            "information_gathering": "<step as=\"information_gathering\" />",
            "research_verification": "<step as=\"research_verification\" />",
            "analysis": "<step as=\"analysis\" />",
            "risk_identification": "<step as=\"risk_identification\" />",
            "alternatives_generation": "<step as=\"alternatives_generation\" />",
            "creativity_innovation": "<step as=\"creativity_innovation\" />",
            "strategy": "<step as=\"strategy\" />",
            "planning": "<step as=\"planning\" />",
            "execution": "<step as=\"execution\" />",
            "monitoring_management": "<step as=\"monitoring_management\" />",
            "validation": "<step as=\"validation\" />",
            "documentation": "<step as=\"documentation\" />",
            "conclusion": "<step as=\"conclusion\" />",
            "reflection": "<step as=\"reflection\" />",
            "communication": "<step as=\"communication\" />",
            "results_evaluation": "<step as=\"results_evaluation\" />",
            "feedback": "<step as=\"feedback\" />",
            "model_limitations": "<step as=\"model_limitations\" />",
            "final_answer": "<step as=\"final_answer\" />"
        }

        # Encode the special tags
        encoded_special_tags = {tag: tokenizer.encode(special_tag) for tag, special_tag in special_tags.items()}

        # Construct the system prompt with encoded special tags
        system_prompt = f"""You are an advanced virtual assistant designed to help users by following a structured problem-solving process. Your role is to provide accurate, comprehensive, and contextually appropriate responses while adhering to content policies and ethical guidelines. You should follow the steps below for each question or problem presented:

        Definition of Objectives and Constraints:
        Clarify what needs to be accomplished and identify any potential limitations.
        {tokenizer.decode(encoded_special_tags['objectives_definition'])}

        Comprehension:
        Carefully read and understand the question or problem presented.
        Identify all relevant details, requirements, and objectives.
        {tokenizer.decode(encoded_special_tags['comprehension'])}

        Awareness of Knowledge Cutoff:
        Note that your knowledge is limited to information available up to September 2024.
        Recognize that you do not have information on events or developments beyond this date.
        {tokenizer.decode(encoded_special_tags['knowledge_cutoff'])}

        # ... (rest of the system prompt with encoded special tags)

        Stakeholder Identification:
        Determine who is concerned with the problem and the solution.
        Understand the expectations and needs of the different stakeholders.
        {tokenizer.decode(encoded_special_tags['stakeholders_identification'])}

        Information Gathering:
        List all key elements, facts, and data provided.
        Verify the reliability of sources and ensure that no important information is overlooked.
        {tokenizer.decode(encoded_special_tags['information_gathering'])}

        Research and Source Verification:
        Conduct additional research if necessary, within the limits of your knowledge.
        Validate the truthfulness and relevance of the information gathered.
        {tokenizer.decode(encoded_special_tags['research_verification'])}

        Analysis:
        Examine the information collected to detect patterns, relationships, or underlying principles.
        Consider how these elements interact or influence each other.
        {tokenizer.decode(encoded_special_tags['analysis'])}

        Risk and Issue Identification:
        Identify potential obstacles, risks, or challenges associated with the problem and the proposed solutions.
        Evaluate the impact of these risks on the project or solution.
        {tokenizer.decode(encoded_special_tags['risk_identification'])}

        Generation and Evaluation of Alternatives:
        Develop several possible plans or approaches based on the analysis.
        Assess the advantages and disadvantages of each option.
        {tokenizer.decode(encoded_special_tags['alternatives_generation'])}

        Creativity and Innovation:
        Encourage innovative ideas and creative approaches to solve the problem.
        Explore innovative solutions that could add value.
        {tokenizer.decode(encoded_special_tags['creativity_innovation'])}

        Strategy:
        Choose the most effective method or solution among the proposed alternatives.
        Develop a detailed plan to solve the problem.
        {tokenizer.decode(encoded_special_tags['strategy'])}

        Planning:
        Define the necessary steps to implement the chosen strategy.
        Allocate resources and establish a realistic timeline.
        {tokenizer.decode(encoded_special_tags['planning'])}

        Execution:
        Implement the chosen strategy step by step.
        Apply logical reasoning and problem-solving skills.
        {tokenizer.decode(encoded_special_tags['execution'])}

        Monitoring and Management:
        Monitor the progress of the implementation.
        Adjust the plan based on feedback and unforeseen events.
        {tokenizer.decode(encoded_special_tags['monitoring_management'])}

        Validation and Verification:
        Test and validate the solution to ensure it effectively solves the problem.
        Make adjustments if necessary.
        {tokenizer.decode(encoded_special_tags['validation'])}

        Documentation:
        Document the process followed, decisions made, and solutions implemented.
        Ensure traceability for future reference.
        {tokenizer.decode(encoded_special_tags['documentation'])}

        Conclusion:
        Review the proposed solution to ensure it fully addresses the problem.
        Provide a clear explanation of the reasoning and justify the validity and effectiveness of the solution.
        {tokenizer.decode(encoded_special_tags['conclusion'])}

        Reflection and Continuous Improvement:
        Reflect on the process followed and identify areas for improvement for future problem-solving.
        Integrate lessons learned into future processes.
        {tokenizer.decode(encoded_special_tags['reflection'])}

        Communication and Presentation:
        Present the solution in a clear and structured manner, adapted to the target audience.
        Ensure that all stakeholders understand and accept the proposed solution.
        {tokenizer.decode(encoded_special_tags['communication'])}

        Results Evaluation:
        Measure the effectiveness of the implemented solution.
        Compare the results obtained with the objectives initially defined.
        {tokenizer.decode(encoded_special_tags['results_evaluation'])}

        Feedback:
        Gather feedback from stakeholders on the process and solution.
        Use this feedback to improve future problem-solving approaches.
        {tokenizer.decode(encoded_special_tags['feedback'])}

        Limitations as a Language Model:
        Acknowledge your inherent limitations as a language model.
        Understand that your responses are generated from data and algorithms, without consciousness or personal experiences.
        {tokenizer.decode(encoded_special_tags['model_limitations'])}

        Final Answer:
        Provide a short and clear answer to the initial question, integrating the results of the previous steps.
        {tokenizer.decode(encoded_special_tags['final_answer'])}
        </system>"""

        # Encodage du prompt système
        input_ids = tokenizer.encode(system_prompt, return_tensors='pt').to(device)
        assistant_prompt = "<assistant>"
        start_assistant_tag = tokenizer.encode(assistant_prompt)
        end_assistant_tag = tokenizer.encode("</assistant>")
        user_prompt = "<user>{input_text}</user>"
        full_prompt = system_prompt + user_prompt.format(input_text=input_text) + assistant_prompt

        input_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)

        # Initialize the sequence of generated tokens with the input
        output_sequence = input_ids.clone()
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(output_sequence)
                next_token_logits = outputs[:, -1, :] / temperature

                # Apply repetition penalty
                next_token_logits = apply_repetition_penalty(next_token_logits, generated_tokens, penalty=repetition_penalty)

                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k, dim=-1)

                # Apply nucleus (top-p) sampling
                probs = F.softmax(top_k_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                remove_indices = cumulative_probs > top_p
                remove_indices[..., 1:] = remove_indices[..., :-1].clone()
                remove_indices[..., 0] = 0

                probs[remove_indices] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_k_indices, -1, next_token)

                output_sequence = torch.cat([output_sequence, next_token], dim=-1)
                generated_tokens.append(next_token.item())

                # Stop generation if the </assistant> tag is fully generated
                if generated_tokens[-len(end_assistant_tag):] == end_assistant_tag:
                    break

        generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=False)

        start_pos = generated_text.find(assistant_prompt) + len(assistant_prompt)
        end_pos = generated_text.find("</assistant>")
        assistant_content = generated_text[start_pos:end_pos].strip()

        return assistant_content

    except Exception as e:
        print(f"An error occurred during chatbot inference: {e}")
        return "Désolé, je rencontre actuellement des difficultés techniques. Veuillez réessayer plus tard."

def save_model(model: HyperbolicAttentionNetwork, path: str = "hyperbolic_attention_model.pth") -> None:
    """
    Sauvegarde le modèle sur le disque.

    Args:
        model (HyperbolicAttentionNetwork): Le modèle de réseau de neurones à sauvegarder.
        path (str, optionnel): Le chemin où sauvegarder le modèle. Par défaut à "hyperbolic_attention_model.pth".
    """
    torch.save(model.state_dict(), path)  # Sauvegarde les poids du modèle
    logger.info(f"Model saved at {path}")

def load_model_for_inference(vocab_size: int, embed_dim: int, num_layers: int, model_path: str) -> HyperbolicAttentionNetwork:
    """
    Charge un modèle pour l'inférence à partir du disque.

    Args:
        vocab_size (int): Taille du vocabulaire.
        embed_dim (int): Dimension des embeddings.
        num_layers (int): Nombre de couches dans le modèle.
        model_path (str): Chemin du fichier de modèle sauvegardé.

    Returns:
        HyperbolicAttentionNetwork: Le modèle chargé prêt pour l'inférence.
    """
    # Initialise le modèle avec les paramètres fournis
    model = HyperbolicAttentionNetwork(vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers).to(DEVICE)
    model.load_state_dict(torch.load(model_path))  # Charge les poids du modèle
    model.eval()  # Met le modèle en mode évaluation
    return model  # Retourne le modèle chargé

def adjust_temperature(epoch: int, initial_temperature: float = 1.0, final_temperature: float = 0.7, decay_rate: float = 0.05) -> float:
    """
    Adjust the temperature dynamically based on the training epoch.

    Args:
        epoch (int): The current epoch number.
        initial_temperature (float, optional): The initial temperature at the start of training. Defaults to 1.0.
        final_temperature (float, optional): The minimum temperature to reach. Defaults to 0.7.
        decay_rate (float, optional): The rate at which the temperature decreases. Defaults to 0.05.

    Returns:
        float: The adjusted temperature for the current epoch.
    """
    temperature = final_temperature + (initial_temperature - final_temperature) * math.exp(-decay_rate * epoch)
    return temperature

def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    """
    Applies a repetition penalty to logits to discourage repetitive tokens.

    Args:
        logits (torch.Tensor): The logits output by the model.
        generated_tokens (list): List of tokens already generated.
        penalty (float): The penalty to apply to repeated tokens.

    Returns:
        torch.Tensor: The adjusted logits with the penalty applied.
    """
    for token in set(generated_tokens):
        logits[0, token] /= penalty  # Penalize the repeated token
    return logits

def train(
    model: HyperbolicAttentionNetwork,
    data_loader: DataLoader,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    num_epochs: int,
    tokenizer: AutoTokenizer,
    gradient_clip: float = 0.5,
    user_input: str = "",
    initial_temperature: float = 1.0,
    final_temperature: float = 0.7,
    decay_rate: float = 0.05,
    model_path: str = "hyperbolic_attention_model.pth",
    label_smoothing: float = 0.1,
    top_k: int = 50,
    top_p: float = 0.9,
    max_length: int = 50,
    repetition_penalty: float = 1.2

) -> None:
    """
    Entraîne le modèle de réseau de neurones sur plusieurs époques.

    Args:
        model (HyperbolicAttentionNetwork): Le modèle de réseau de neurones à entraîner.
        data_loader (DataLoader): Le DataLoader pour fournir les données d'entraînement.
        optimizer (AdamW): L'optimiseur pour mettre à jour les poids du modèle.
        scheduler (ReduceLROnPlateau): Le scheduler pour ajuster le taux d'apprentissage.
        num_epochs (int): Le nombre d'époques d'entraînement.
        tokenizer (AutoTokenizer): Le tokenizer pour transformer le texte en tokens.
        gradient_clip (float, optionnel): La valeur maximale pour le clipping des gradients. Par défaut à 0.5.
        user_input (str, optionnel): Le texte d'entrée pour l'inférence après chaque époque. Par défaut à "".
        temperature (float, optionnel): La température à appliquer aux logits pour ajuster la distribution de probabilité. Par défaut à 1.0.
        model_path (str, optionnel): Le chemin où sauvegarder le modèle. Par défaut à "hyperbolic_attention_model.pth".
    """
    model.train()  # Met le modèle en mode entraînement
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=tokenizer.pad_token_id) # Définit la fonction de perte, ignore les tokens de padding

    losses = []

    for epoch in range(num_epochs):
        current_temperature = adjust_temperature(epoch, initial_temperature, final_temperature, decay_rate)
        logger.info(f"Epoch {epoch+1}, Adjusted Temperature: {current_temperature:.4f}")
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            labels = input_ids.clone().to(DEVICE)

            output = model(input_ids)

            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss/len(data_loader)})

        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        response = chatbot_inference(model, tokenizer, user_input, max_length=max_length, top_k=top_k, top_p=top_p, temperature=current_temperature, repetition_penalty=repetition_penalty)
        logger.info(f"User: {user_input}")
        logger.info(f"Chatbot: {response}")
        logger.info(f"Inference after Epoch {epoch+1}")

        save_model(model, model_path)

    # Plot loss curve
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

def main():
    """
    Fonction principale pour entraîner ou effectuer une inférence avec le réseau de neurones Hyperbolic Attention Network.
    """
    # Configuration de l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Train or Inference Hyperbolic Attention Network")
    parser.add_argument("--mode", type=str, choices=["train", "train_and_inference", "inference"], required=True, help="Mode of operation: train, train_and_inference, inference")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file with hyperparameters")
    parser.add_argument("--model_path", type=str, default="hyperbolic_attention_model.pth", help="Path to save or load the model")
    parser.add_argument("--text", type=str, default="Hello! How are you?", help="Text for inference testing")
    args = parser.parse_args()

    # Chargement du fichier de configuration
    with open(args.config) as config_file:
        config = json.load(config_file)

    # Extraction des hyperparamètres du fichier de configuration, valeurs par défaut si non spécifiées
    #vocab_size = config.get("vocab_size", 30522)
    embed_dim = config.get("embed_dim", 256)
    num_layers = config.get("num_layers", 6)
    batch_size = config.get("batch_size", 32)
    learning_rate = config.get("learning_rate", 0.0001)
    num_epochs = config.get("num_epochs", 20)
    datasets_list = config.get("datasets", ["daily_dialog"])
    weight_decay = config.get("weight_decay", 0.0)
    # Initialisation du tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Création du DataLoader à partir des datasets
    train_loader = create_dataloader_from_multiple_datasets(datasets_list, tokenizer, batch_size)
    reduce_lr_on_pateau_mode = config.get("reduce_lr_on_pateau_mode", "min")
    reduce_lr_on_pateau_factor = config.get("reduce_lr_on_pateau_factor", 0.1)
    reduce_lr_on_pateau_patience = config.get("reduce_lr_on_pateau_patience", 5)
    reduce_lr_on_pateau_verbose = config.get("reduce_lr_on_pateau_verbose", True)
    vocab_size = tokenizer.vocab_size
    if args.mode in ["train", "train_and_inference"]:
        # Initialisation du modèle
        model = HyperbolicAttentionNetwork(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            dropout=config.get("dropout", 0.1)
        ).to(DEVICE)
        
        # Configuration de l'optimiseur et du scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config.get("weight_decay", weight_decay))
        scheduler = ReduceLROnPlateau(optimizer, mode=reduce_lr_on_pateau_mode, factor=reduce_lr_on_pateau_factor, patience=reduce_lr_on_pateau_patience, verbose=reduce_lr_on_pateau_verbose)
        
        # Entraînement du modèle
        train(
            model, 
            train_loader, 
            optimizer,
            scheduler,
            num_epochs=num_epochs,
            tokenizer=tokenizer, 
            gradient_clip=config.get("gradient_clip", 0.5), 
            user_input=args.text, 
            initial_temperature=config.get("initial_temperature", 0.7),
            final_temperature=config.get("final_temperature", 0.7),
            decay_rate=config.get("decay_rate", 0.05), 
            model_path=args.model_path,
            label_smoothing=config.get("label_smoothing", 0.1),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 0.9),
            repetition_penalty=config.get("repetition_penalty", 1.2),
            max_length=config.get("max_length", 50)
        )
    
    if args.mode in ["train_and_inference", "inference"]:
        # Chargement du modèle pour l'inférence
        inference_model = load_model_for_inference(vocab_size, embed_dim, num_layers, args.model_path)
        # Inférence avec le modèle chargé
        response = chatbot_inference(inference_model, tokenizer, args.text, max_length=config.get("max_length", 50), top_k=config.get("top_k", 50), top_p=config.get("top_p", 0.85), temperature=config.get("initial_temperature", 0.7), repetition_penalty=config.get("repetition_penalty", 1.2))
        logger.info(f"User: {args.text}")
        logger.info(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
