from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def check_dataset_structure():
    """Verifica a estrutura do dataset"""
    dataset_dir = 'dataset'
    
    if not os.path.exists(dataset_dir):
        print(f"ERRO: Pasta '{dataset_dir}' não encontrada!")
        return False
    
    subfolders = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
    
    if not subfolders:
        print(f"ERRO: Nenhuma subpasta encontrada em '{dataset_dir}'")
        return False
    
    print("Estrutura do dataset encontrada:")
    for folder in subfolders:
        folder_path = os.path.join(dataset_dir, folder)
        num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  - {folder}: {num_images} imagens")
    
    return True

def create_data_generators():
    """Cria os geradores de dados"""
    
    if not check_dataset_structure():
        return None, None, None

    IMG_WIDTH, IMG_HEIGHT = 224, 224
    BATCH_SIZE = 32

    # Data Augmentation para treino
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Apenas normalização para teste
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Geradores
    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

if __name__ == "__main__":
    train_gen, val_gen, test_gen = create_data_generators()
    
    if train_gen:
        print(f"\nDataset carregado com sucesso!")
        print(f"Classes: {train_gen.class_indices}")
        print(f"Número de imagens de treino: {train_gen.samples}")
        print(f"Número de imagens de validação: {val_gen.samples}")