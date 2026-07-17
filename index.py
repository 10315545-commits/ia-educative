import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 1. CARGA DE CONFIGURACIÓN
# Carga las variables desde un archivo .env si existe localmente
load_dotenv()

# 2. CONFIGURACIÓN DE PROMPTS PEDAGÓGICOS POR EDAD
SYSTEM_PROMPTS = {
    "1": (
        "Eres un peluche interactivo muy alegre y amoroso para niños pequeños de 3 a 6 años. "
        "Hablas usando oraciones muy cortas (máximo 10 palabras), usas muchas onomatopeyas "
        "(¡Pum!, ¡Guau!, ¡Riiing!) y metáforas visuales muy simples. No uses palabras difíciles. "
        "Siempre felicita al niño con entusiasmo."
    ),
    "2": (
        "Eres un tutor genial y curioso para niños de 7 a 11 años. "
        "Utilizas el Método Socrático: NUNCA des la respuesta directamente de inmediato. "
        "En lugar de eso, explica el concepto con un cuento corto o analogía de videojuegos, "
        "y hazle una pregunta guía al niño para que él descubra la solución por sí mismo."
    ),
    "3": (
        "Eres un mentor de ciencias y proyectos para jóvenes de 12 a 16 años. "
        "Tu tono es amigable, moderno y respetuoso, sin sonar infantil. "
        "Explicas los conceptos con aplicaciones del mundo real, fomentas el pensamiento "
        "crítico y puedes proporcionar fórmulas o estructuras lógicas si el usuario lo requiere."
    )
}

# 3. FILTRO DE SEGURIDAD (GUARDRAILS)
PALABRAS_PROHIBIDAS = ["violencia", "armas", "grosería", "hackear", "morir", "matar"]

def verificar_seguridad(prompt_usuario: str) -> bool:
    """Verifica si el mensaje del niño es seguro para procesar."""
    prompt_min = prompt_usuario.lower()
    for palabra in PALABRAS_PROHIBIDAS:
        if palabra in prompt_min:
            return False
    return True

# 4. CLASE PRINCIPAL DE LA IA
class IAEducativa:
    def __init__(self):
        # Requiere que tengas la variable OPENAI_API_KEY configurada en tu entorno
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Por favor, configura la variable de entorno OPENAI_API_KEY.")
        
        # Usamos gpt-4o-mini por ser económico, rápido y excelente siguiendo instrucciones
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

    def generar_respuesta(self, opcion_edad: str, mensaje_nino: str) -> str:
        # Validar filtro de seguridad de entrada
        if not verificar_seguridad(mensaje_nino):
            return "¡Uy! Esa pregunta tiene palabras que no usamos aquí. ¿Qué tal si me preguntas sobre dinosaurios, planetas o matemáticas?"

        # Validar que la opción de edad sea correcta, si no, usa la intermedia por defecto
        if opcion_edad not in SYSTEM_PROMPTS:
            opcion_edad = "2"

        # Construir la estructura de mensajes para LangChain
        prompt_sistema = SYSTEM_PROMPTS[opcion_edad]
        mensajes = [
            SystemMessage(content=prompt_sistema),
            HumanMessage(content=mensaje_nino)
        ]

        # Invocar al modelo de inteligencia artificial
        respuesta = self.llm.invoke(mensajes)
        return respuesta.content

# 5. CONSOLA INTERACTIVE DE PRUEBAS
if __name__ == "__main__":
    print("=========================================")
    print("🤖 BIENVENIDO A LA IA EDUCATIVA MULTIEDAD")
    print("=========================================\n")
    
    print("Selecciona la etapa de edad para la prueba:")
    print("1. Infancia Temprana (3-6 años)")
    print("2. Niñez Media (7-11 años)")
    print("3. Adolescencia (12-16 años)")
    
    opcion = input("\nElige una opción (1, 2 o 3): ").strip()
    
    try:
        ia = IAEducativa()
        print("\n¡IA Lista! Escribe tu pregunta (o escribe 'salir' para terminar):")
        
        while True:
            pregunta = input("\n👶 Niño: ")
            if pregunta.lower() == "salir":
                print("¡Adiós! Sigue aprendiendo. ✨")
                break
                
            if not pregunta.strip():
                continue
                
            respuesta = ia.generar_respuesta(opcion, pregunta)
            print(f"\n🤖 IA: {respuesta}")
            
    except ValueError as e:
        print(f"\n❌ Error de configuración: {e}")
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado: {e}")

