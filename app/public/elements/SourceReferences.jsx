import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";

export default function SourceReferences() {
  const sources = props.sources || [];

  if (!sources.length) {
    return null;
  }

  const openSource = async (source) => {
      if (source.url) {
          window.open(source.url, "_blank", "noopener,noreferrer");
      }

      try {
          await callAction({
              name: "open_source",
              payload: {
                  source_id: source.id,
              },
          });
      } catch (error) {
          console.error("Failed to call open_source action", error);
      }
  };

  return (
    <div className="mt-4 flex flex-wrap gap-2">
      {sources.map((source) => (
        <Button
          key={source.id}
          variant="outline"
          size="sm"
          onClick={() => openSource(source)}
        >
          <ExternalLink className="mr-2 h-4 w-4" />
          {source.label}
        </Button>
      ))}
    </div>
  );
}
