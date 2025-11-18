from crop_wells import crop_wells

def main():
    file_path = "data/IBV/Plate_1.jpg"
    plate_type = "IBV"

    crop_wells(
        file_path=file_path,
        plate_type=plate_type,
        debug=True,
        save_files=True
    )

if __name__ == "__main__":
    main()