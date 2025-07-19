declare module "react-leaflet-draw" {
  import { Component } from "react";
  import { Control } from "leaflet";

  interface DrawEventLayer {
    layerType: string;
    layer: {
      getLatLngs: () => Array<Array<{ lat: number; lng: number }>>;
      [key: string]: unknown;
    };
  }

  export interface EditControlProps {
    position?: Control.Position;
    onEdited?: (e: unknown) => void;
    onCreated?: (e: DrawEventLayer) => void;
    onDeleted?: (e: unknown) => void;
    draw?: Record<string, unknown>;
    edit?: Record<string, unknown>;
  }

  export class EditControl extends Component<EditControlProps> {}
}
