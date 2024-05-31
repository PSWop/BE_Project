import os
import sys
from pydub import AudioSegment


def chunk_wav(file_name, chunk_duraction=0.5):
    """Chunk a wav file into 0.5 second segments and save them in a folder
    :param file_path (str): Path to wav file
    :param chunk_duraction (float): Duration of each segment in seconds (default is 0.5)
    :return:
    """
    # Create a directory to save the chunk files
    output_directory = f"chunks/{file_name}"[:-4]
    os.makedirs(output_directory, exist_ok=True)

    # Read the input WAV file
    song = AudioSegment.from_wav(f"songs/{file_name}")

    # Calculate the length of each chunk (0.5 seconds)
    chunk_length = 500  # milliseconds

    # Split the song into chunks and save each chunk
    for i, chunk_start in enumerate(range(0, len(song), chunk_length)):
        chunk_end = chunk_start + chunk_length
        chunk = song[chunk_start:chunk_end]

        # Create the output file path
        chunk_output_path = os.path.join(output_directory, f"chunk_{i}.wav")

        # Export the chunk as a WAV file
        chunk.export(chunk_output_path, format="wav")

    print(f"Song successfully split into {i + 1} chunks.")


def main():
    try:
        song_name = sys.argv[-1]
        chunk_wav(song_name)
    except:
        print("Please provide a correct song name")


if __name__ == "__main__":
    main()
