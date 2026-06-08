import { ImageUp } from "lucide-react";

function ImageUploader({ imageFile, previewUrl, onImageChange, disabled }) {
  const handleChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      onImageChange(file);
    }
  };

  return (
    <section className="panelBlock">
      <div className="panelHeader">
        <ImageUp size={18} />
        <h2>Image</h2>
      </div>

      <label className={`uploadBox ${disabled ? "disabled" : ""}`}>
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          disabled={disabled}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="Uploaded vehicle preview" />
        ) : (
          <span>Choose vehicle image</span>
        )}
      </label>

      <div className="fileMeta">
        {imageFile ? imageFile.name : "No image selected"}
      </div>
    </section>
  );
}

export default ImageUploader;
