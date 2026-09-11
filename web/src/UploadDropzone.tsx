import React, { useState } from "react";
import { Upload } from "lucide-react";

/** The formats the canonicaliser accepts, and the only place that says so. */
export const ACCEPTED_FILE_EXTENSIONS = [".txt", ".docx", ".doc", ".pdf"];

export function getFileExtension(name: string): string {
  const dotIndex = name.lastIndexOf(".");
  return dotIndex !== -1 ? name.slice(dotIndex).toLowerCase() : "";
}

function formatMegabytes(bytes: number): string {
  return new Intl.NumberFormat("pl-PL", {
    style: "unit",
    unit: "megabyte",
    maximumFractionDigits: 0,
  }).format(bytes / (1024 * 1024));
}

interface UploadDropzoneProps {
  onFileSelected: (file: File) => void;
  maxInputBytes?: number | null;
  label: string;
}

/**
 * The control that takes a contract from the reader, in both places that offer
 * one: the empty workspace, and an expired session that needs a fresh upload to
 * carry on. Drag state is local because nothing outside this control reads it.
 */
export const UploadDropzone: React.FC<UploadDropzoneProps> = ({
  onFileSelected,
  maxInputBytes,
  label,
}) => {
  const [isDragging, setIsDragging] = useState(false);

  return (
    <label
      className={`upload-dropzone${isDragging ? " drag-over" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        const file = event.dataTransfer.files[0];
        if (file) {
          onFileSelected(file);
        }
      }}
    >
      <Upload size={36} className="upload-icon" />
      <p className="upload-prompt">{isDragging ? "Upuść plik tutaj" : label}</p>
      <p className="upload-formats">
        Obsługiwane formaty: {ACCEPTED_FILE_EXTENSIONS.join(", ")}
        {maxInputBytes ? ` (maks. ${formatMegabytes(maxInputBytes)})` : ""}
      </p>
      <input
        className="visually-hidden"
        type="file"
        aria-label="Wybierz plik umowy"
        data-testid="file-upload-input"
        accept={ACCEPTED_FILE_EXTENSIONS.join(",")}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            event.currentTarget.click();
          }
        }}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            onFileSelected(file);
          }
        }}
      />
    </label>
  );
};
