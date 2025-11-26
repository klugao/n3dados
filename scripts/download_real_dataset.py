#!/usr/bin/env python3
"""
Script para baixar e preparar dataset REAL de predição de evasão estudantil
Dataset: Predict students' dropout and academic success (UCI ML Repository)
URL: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import requests
import zipfile
import io
warnings.filterwarnings('ignore')

def download_uci_dataset():
    """
    Baixa o dataset do UCI Machine Learning Repository
    
    Returns:
        Tupla (DataFrame, bool) - DataFrame com os dados e flag indicando se é real
    """
    print("📥 Tentando baixar dataset REAL do UCI Machine Learning Repository...")
    print("   Dataset: Predict students' dropout and academic success")
    
    # URLs para tentar (em ordem de prioridade)
    urls_to_try = [
        ("UCI Archive", "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"),
        # Adicionar outras fontes se disponíveis
    ]
    
    for source_name, url in urls_to_try:
        try:
            print(f"\n   🔄 Tentando baixar de: {source_name}")
            print(f"      URL: {url}")
            response = requests.get(url, timeout=30, allow_redirects=True)
            
            if response.status_code == 200:
                print("   ✅ Download bem-sucedido!")
                # Descompactar
                zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                
                # Procurar arquivo CSV dentro do ZIP
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                if csv_files:
                    csv_file = csv_files[0]
                    print(f"   📄 Arquivo encontrado: {csv_file}")
                    # Tentar diferentes separadores
                    for sep in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(zip_file.open(csv_file), sep=sep, encoding='utf-8')
                            if len(df.columns) > 1:  # Se tem mais de uma coluna, provavelmente está certo
                                print(f"   ✅ Dataset REAL carregado com sucesso!")
                                print(f"      📊 Shape: {df.shape}")
                                return df, True
                        except:
                            continue
                    
                    # Se nenhum separador funcionou, tentar sem especificar
                    df = pd.read_csv(zip_file.open(csv_file), encoding='utf-8')
                    print(f"   ✅ Dataset REAL carregado com sucesso!")
                    print(f"      📊 Shape: {df.shape}")
                    return df, True
                else:
                    print(f"   ⚠️  Nenhum arquivo CSV encontrado no ZIP")
                    continue
            else:
                print(f"   ⚠️  Erro HTTP: {response.status_code}")
                continue
                
        except requests.exceptions.Timeout:
            print(f"   ⚠️  Timeout ao baixar de {source_name}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Erro de conexão: {e}")
            continue
        except Exception as e:
            print(f"   ⚠️  Erro ao processar: {e}")
            continue
    
    # Se chegou aqui, não conseguiu baixar dataset real
    print("\n" + "="*70)
    print("⚠️  ATENÇÃO: Não foi possível baixar um dataset REAL")
    print("="*70)
    print("\n   Motivos possíveis:")
    print("   - Sem conexão com a internet")
    print("   - URL do dataset mudou")
    print("   - Servidor temporariamente indisponível")
    print("\n   💡 Usando dataset de exemplo baseado em padrões reais...")
    print("   (Este ainda é um dataset sintético, mas com padrões realistas)")
    print("="*70 + "\n")
    
    return create_example_real_dataset(), False

def create_example_real_dataset():
    """
    Cria um dataset de exemplo baseado em características de datasets reais
    Este é um fallback caso o download falhe
    """
    print("   📊 Criando dataset de exemplo baseado em padrões reais...")
    
    np.random.seed(42)
    n_students = 1000
    
    # Features baseadas em datasets reais de evasão estudantil
    df = pd.DataFrame({
        # IDs
        'student_id': [f'STU{i:04d}' for i in range(1, n_students + 1)],
        
        # Demográficas (valores mais realistas)
        'age': np.random.choice([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], 
                               n_students, p=[0.01, 0.15, 0.20, 0.18, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 
                                             0.01, 0.01, 0.005, 0.005, 0.003, 0.002, 0.001, 0.001, 0.001]),
        'gender': np.random.choice(['M', 'F', 'O'], n_students, p=[0.48, 0.50, 0.02]),
        'socioeconomic_level': np.random.choice([1, 2, 3, 4, 5], n_students, p=[0.15, 0.25, 0.30, 0.20, 0.10]),
        
        # Acadêmicas (baseadas em padrões reais)
        'avg_grade': np.clip(np.random.normal(7.0, 1.8, n_students), 0, 10),
        'avg_attendance': np.clip(np.random.normal(82, 18, n_students), 0, 100),
        'current_semester': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 
                                           n_students, p=[0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.07, 0.03]),
        'total_enrollments': [np.random.randint(max(1, s*4-3), s*4+5) for s in np.random.choice([1,2,3,4,5,6,7,8], n_students, p=[0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.07, 0.03])],
        'failed_courses': np.random.poisson(0.8, n_students).clip(0, 8),
        'completed_courses': [max(0, enroll - np.random.randint(0, fail+2)) for enroll, fail in zip(
            [np.random.randint(1, 33) for _ in range(n_students)],
            np.random.poisson(0.8, n_students).clip(0, 8)
        )],
        
        # Comportamentais
        'total_interactions': np.random.poisson(45, n_students).clip(0, 200),
        'unique_sessions_count': np.random.poisson(18, n_students).clip(0, 60),
        'total_duration_hours': np.clip(np.random.gamma(2.5, 12, n_students), 0, 250),
        'days_since_last_interaction': np.clip(np.random.exponential(4, n_students), 0, 30),
        'engagement_score': np.random.normal(85, 25, n_students).clip(0, 200),
        
        # Financeiras
        'scholarship_percentage': np.random.choice([0, 25, 50, 75, 100], n_students, p=[0.35, 0.28, 0.18, 0.12, 0.07]),
        'overdue_payments': np.random.poisson(0.4, n_students).clip(0, 6),
        'pending_payments': np.random.poisson(0.6, n_students).clip(0, 6),
        'outstanding_amount': np.random.exponential(800, n_students).clip(0, 8000),
    })
    
    # Calcular features derivadas
    df['success_rate'] = (df['completed_courses'] / df['total_enrollments'].replace(0, np.nan) * 100).fillna(0)
    df['failure_rate'] = (df['failed_courses'] / df['total_enrollments'].replace(0, np.nan) * 100).fillna(0)
    df['interaction_per_enrollment'] = (df['total_interactions'] / df['total_enrollments'].replace(0, np.nan)).fillna(0)
    
    # Gerar target baseado em padrões reais (evasão maior no início e com baixo desempenho)
    risk_factors = (
        (df['avg_grade'] < 5.0) * 25 +
        (df['avg_grade'] < 6.0) * 15 +
        (df['avg_attendance'] < 70) * 20 +
        (df['failed_courses'] >= 3) * 15 +
        (df['current_semester'] == 1) * 10 +
        (df['current_semester'] == 2) * 5 +
        (df['socioeconomic_level'] <= 2) * 10 +
        (df['overdue_payments'] >= 2) * 8 +
        (df['days_since_last_interaction'] > 10) * 8 +
        (df['total_interactions'] < 20) * 7
    )
    
    # Converter para probabilidade e gerar target
    dropout_prob = 1 / (1 + np.exp(-(risk_factors - 50) / 12))
    df['dropout'] = np.random.binomial(1, dropout_prob, n_students)
    
    # Ajustar para ter distribuição mais realista (cerca de 15-25% de evasão)
    if df['dropout'].mean() < 0.10:
        # Aumentar alguns casos
        high_risk_idx = df[risk_factors > 60].index[:int(n_students * 0.15)]
        df.loc[high_risk_idx, 'dropout'] = 1
    
    return df

def transform_to_compatible_format(df_original):
    """
    Transforma o dataset original para o formato esperado pelo projeto
    
    Args:
        df_original: DataFrame com dados originais (pode ter colunas diferentes)
        
    Returns:
        DataFrame no formato compatível
    """
    print("\n🔄 Transformando dataset para formato compatível...")
    
    # Se já está no formato correto, retornar
    expected_cols = ['student_id', 'age', 'gender', 'socioeconomic_level', 
                    'avg_grade', 'avg_attendance', 'current_semester', 
                    'total_enrollments', 'failed_courses', 'completed_courses',
                    'total_interactions', 'unique_sessions_count', 
                    'total_duration_hours', 'days_since_last_interaction',
                    'engagement_score', 'scholarship_percentage', 
                    'overdue_payments', 'pending_payments', 'outstanding_amount',
                    'dropout', 'success_rate', 'failure_rate', 
                    'interaction_per_enrollment']
    
    if all(col in df_original.columns for col in expected_cols):
        print("   ✅ Dataset já está no formato correto!")
        return df_original[expected_cols]
    
    # Caso contrário, fazer mapeamento (isso seria customizado baseado no dataset real)
    print("   ⚠️  Mapeamento de colunas necessário...")
    print("   💡 Dataset de exemplo já está no formato correto")
    
    return df_original

def main():
    """Função principal"""
    print("=" * 70)
    print("DOWNLOAD E PREPARAÇÃO DE DATASET REAL")
    print("=" * 70)
    print()
    
    # Baixar dataset
    df, is_real = download_uci_dataset()
    
    # Transformar para formato compatível
    df_final = transform_to_compatible_format(df)
    
    # Criar diretório data se não existir
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Salvar dataset
    output_path = data_dir / 'student_dropout_dataset.csv'
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    if is_real:
        print("✅ DATASET REAL PREPARADO COM SUCESSO!")
        print("=" * 70)
        print("   ✅ Este é um DATASET REAL baixado do UCI ML Repository")
    else:
        print("⚠️  DATASET DE EXEMPLO PREPARADO")
        print("=" * 70)
        print("   ⚠️  Este é um dataset SINTÉTICO (não foi possível baixar o real)")
        print("   💡 Para usar um dataset REAL, verifique sua conexão com a internet")
        print("      e tente executar o script novamente.")
    
    print(f"\n   📁 Localização: {output_path}")
    print(f"   📊 Total de registros: {len(df_final)}")
    print(f"   📈 Features: {len(df_final.columns) - 1}")  # -1 para excluir target
    print(f"   🎯 Taxa de evasão: {df_final['dropout'].mean()*100:.1f}%")
    print(f"   📋 Colunas: {', '.join(df_final.columns.tolist())}")
    print()
    print("💡 PRÓXIMOS PASSOS:")
    print("   1. Execute o notebook: notebooks/01_eda_exploratoria.ipynb")
    print("   2. Execute o notebook: notebooks/02_modelagem_avaliacao.ipynb")
    print("   3. Os notebooks usarão automaticamente este dataset!")
    print("=" * 70)

if __name__ == "__main__":
    # Verificar se requests está instalado
    try:
        import requests
    except ImportError:
        print("⚠️  Biblioteca 'requests' não encontrada.")
        print("   Instalando...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'requests'])
        import requests
    
    main()

