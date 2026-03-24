import os


def save_captions(text_content, output_dir, filename="captions.txt"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    full_path = os.path.join(output_dir, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Captions saved to: {full_path}")
    return full_path

# try run:
# if __name__ == "__main__":
#     save_captions()
