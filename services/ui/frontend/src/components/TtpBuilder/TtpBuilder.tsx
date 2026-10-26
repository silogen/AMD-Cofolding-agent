// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

export interface CreatedTtp {
  id: string;
  name: string;
}

interface Props {
  onClose: () => void;
  /** Called once create_ttp has stored the profile. */
  onCreated: (ttp: CreatedTtp) => void;
}

/** A form to define a target product profile without asking the agent. */
export function TtpBuilder({ onClose }: Props) {
  return (
    <div role="dialog" aria-label="New target product profile" className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-raised border-subtle rounded-lg border p-4 text-sm">
        Coming soon.
        <button type="button" onClick={onClose} className="ml-3 cursor-pointer underline">
          Close
        </button>
      </div>
    </div>
  );
}
